(ns assignment.r-caret
  (:require
    [assignment.eda :refer [concrete-data]]
    [calc-metric.patch]
    [clojisr.v1.applications.plotting
     :refer [plot->svg]]
    [clojisr.v1.r :refer [bra r+ r-]]
    [clojisr.v1.require :refer [require-r]]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.ml.dataset :as ds]))

(comment
  (clojure.java.shell/sh "which" "R"))

; # Caret with R Interop
; In this chapter, I will graph a tuned decision tree. The issue with Smile's implementation is that there is no `.dot` method of class `gradient.tree` like there is with the classification tree in the [scicloj models tutorial](https://scicloj.github.io/scicloj.ml-tutorials/userguide-models.html).
;
; In native Clojure, I tried the scicloj Kroki method, but failed. So I turned to my trusty R... in Clojure. While R has a nice plotting function for tree models, viz. `prp` and `rpart.plot`, neither of them returned a plottable output. Instead they returned the data structure that would, I suppose, pass to a graphing library.
;
; Because I find `prp` and `rpart.plot` outputs beautiful, I tried to plot them in Clojure with R's `Rgraphviz`, `igraph`, and `tree.data` libraries, all to no success.

; ## Load the required R libraries
(require-r '[base :refer [RNGkind set-seed summary plot $ which-min]]
           '[stats :refer [predict]]
           '[caret :refer [createDataPartition trainControl modelLookup
                           train defaultSummary postResample]]
           '[rpart :refer [rpart]]
           '[rpart.plot :refer [rpart-plot prp]]
           '[ggplot2 :refer [ggplot aes geom_segment geom_text scale_size]]
           '[ggdendro :refer [dendro_data theme_dendro segment
                              label leaf_label]])

(summary concrete-data)

; ## Setup dataset
; The R interop does not like Clojure's convention of columns as keyword (with their prefixed colons) nor the kabab-casing. Below is my assisting Clojure-R op to work with the data.
(def r-data
  (ds/rename-columns concrete-data (fn [col]
                                     (-> col
                                         name               ; removes ":" from type keyword
                                         (clojure.string/replace #"-" ".")))))

;; ## Partition data
(def index
  (createDataPartition :y (:concrete-compressive-strength concrete-data)
                       :p 0.7 :list false))
(def training-data
  (bra r-data index nil))
(def test-data
  (bra r-data (r- index) nil))

; ## Caret decision tree
(RNGkind :sample.kind "Rounding")
(set-seed 0)

;; ### Bootstrap cross-validation
(def train-control
  (trainControl :method "boot" :number 25))

(modelLookup "rpart")

;; ### Build model
(def decision-tree
  (train '(tilde concrete.compressive.strength
                 (+ cement blast.furnace.slag fly.ash water
                    superplasticizer coarse.aggregate fine.aggregate
                    age concrete.compressive.strength)) :data training-data
         :method "rpart" :trControl train-control :metric "MAE" :tuneLength 20))

decision-tree

^kind/hiccup
(-> (plot decision-tree)
    plot->svg)

($ decision-tree 'results)

;; ### Best hyperparameter
(-> ($ decision-tree 'results)
    (bra (which-min
           (bra ($ decision-tree 'results) nil 4))
         nil))

;; ### View tree
($ decision-tree 'finalModel)

;; ## Plot tree
(def plot-tree
  (rpart-plot ($ decision-tree 'finalModel)
              :type 1 :extra 1 :under true :split.font 2 :varlen -10))

(def ddata
  (dendro_data ($ decision-tree 'finalModel)))

^kind/hiccup
(-> (ggplot)
    (r+ (geom_segment :data ($ ddata 'segments)
                      (aes :x 'x :y 'y :xend 'xend :yend 'yend)))
    (r+ (geom_text :data ($ ddata 'labels)
                   (aes :x 'x :y 'y :label 'label)
                   :size 3 :vjust -0.5))
    (r+ (geom_text :data ($ ddata 'leaf_labels)
                   (aes :x 'x :y 'y :label 'label)
                   :size 3 :vjust 1))
    (r+ (theme_dendro))
    plot->svg)

; Truly an ugly tree. As I said above, `prp` and `rpart.plot` do a much better job, but if you see the `plot-tree` variable defined above, you'd see that it is only a data structure, not a plot as it would be in R.

^kind/hiccup
(-> (ggplot (segment ddata))
    (r+ (geom_segment (aes :x 'x :y 'y :size 'n :xend 'xend :yend 'yend)
                      :color "blue" :alpha 0.5))
    (r+ (geom_text :data (label ddata)
                   (aes :x 'x :y 'y :label 'label)
                   :size 3 :vjust -0.5))
    (r+ (geom_text :data (leaf_label ddata)
                   (aes :x 'x :y 'y :label 'label)
                   :size 3 :vjust 1))
    (r+ (theme_dendro))
    plot->svg)

; This plot is almost worse in terms of good looks. But there was a [vignette](https://cran.r-project.org/web/packages/ggdendro/vignettes/ggdendro.html) of it and I tried sooo many other ways of plotting a tree, I had to include it.

; ## Evaluate model
(def pred
  (predict ($ decision-tree 'finalModel) test-data))

(postResample pred ($ test-data 'concrete.compressive.strength))

; Somewhat surprisingly, Smile did much better in terms of R$^2$. However, the amount of hyperparameters a gradient tree can take is much more than the rpart tree. Not to mention, how wide I let the grid be. Realistically, I would higher the `node-size` as to not allow more depth as to lower variance. The MAE and RMSE metrics are almost the same.