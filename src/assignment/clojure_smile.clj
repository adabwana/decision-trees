(ns assignment.clojure-smile
  (:require
    [assignment.eda :refer [concrete-data]]
    [calc-metric.patch]
    [aerial.hanami.templates :as ht]
    [scicloj.noj.v1.vis.hanami :as hanami]
    [fastmath.stats :as stats]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.metamorph.ml.viz :as ml-viz]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]))

;; # Smile with Clojure
;; ## Define regressors and response
(def response :concrete-compressive-strength)
(def regressors
  (ds/column-names concrete-data (complement #{response})))

(ds/info concrete-data)

;; ## Setup pipelines
; This is the data to model machine. Here I would add normalizing, standardizing, feature engineering, etc. I chose to leave it plain for simplicity. I plan to add parameters in `create-model-pipeline` in a future assignment to accommodate different data processing machines.
(def pipeline-fn
  (ml/pipeline
    (mm/set-inference-target response)))

;; ### Generic pipeline function
; Abstracted my process since last assignment and continuing that process.
(defn create-model-pipeline
  [model-type params]
  (ml/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (mm/model (merge {:model-type model-type} params))))

;; #### Gradient tree context
; This is not a "regression tree" as typically though in R's `rpart` library. Clojure's Smile does not have any regression tree models. The closest thing I found was a gradient tree. Below, in the grid-search code, you will see I set one of it's `:trees` hyperparameters to 1 as to make a single "regression tree."
(defn gradient-tree-pipe-fn
  [params]
  (create-model-pipeline :smile.regression/gradient-tree-boost params))

;; #### Random forest context
(defn random-forest-pipe-fn
  [params]
  (create-model-pipeline :smile.regression/random-forest params))

;; ## Pipeline Functions
; Last section shows the data processing, generic pipeline, and context specific pipeline all as functions. Perhaps this functions fed to functions is how many programming languages can/should be passed, I only found it fun and interesting in Clojure. It looks and feels just like defining a variable, no fancy curly brackets for R functions and no syntactic colons for Python.
;; ### Evaluate pipeline
(defn evaluate-pipe [pipe data]
  (ml/evaluate-pipelines
    pipe
    data
    stats/omega-sq
    :accuracy
    {:other-metrices                   [{:name :mae :metric-fn ml/mae}
                                        {:name :rmse :metric-fn ml/rmse}]
     :return-best-pipeline-only        false
     :return-best-crossvalidation-only true}))

;; ### Generate hyperparameters for models
; This was an interesting section if only because scicloj.ml.metamorph library's hyperparameters 1) didn't include the hyperparameters for some of its models and 2) can return the wrong type for the hyperparameters. You can find model definitions and some examples at scicloj's model [tutorial](https://scicloj.github.io/scicloj.ml-tutorials/userguide-models.html).

; No hyperparameters with :smile.regression/gradient-tree-boost or :smile.regression/random-forest

(ml/hyperparameters :smile.classification/decision-tree)

; Look at the types for the hyperparameters for a classification tree. For example, `max-nodes`, why is the type `:float64`? Are we going to have 3.7 nodes? To make the gridsearch, I had to refer to the tutorial above for both the hyperparameters, types, and default grid ranges.

(defn generate-hyperparams [model-type]
  (case model-type
    :gradient-tree (take 60
                         (ml/sobol-gridsearch {:trees       1 ;(ml/linear 1 1000 10 :int32)
                                               :loss        (ml/categorical [:least-absolute-deviation :least-squares])
                                               :max-depth   (ml/linear 10 50 20 :int32)
                                               :max-nodes   (ml/linear 10 1000 30 :int32)
                                               :node-size   (ml/linear 1 20 20 :int32)
                                               :shrinkage   (ml/linear 0.1 1)
                                               :sample-rate (ml/linear 0.1 1 10)}))
    :random-forest (take 30
                         (ml/sobol-gridsearch {:trees       (ml/linear 1 1000 10 :int32)
                                               :max-depth   (ml/linear 10 50 20 :int32)
                                               :max-nodes   (ml/linear 10 1000 30 :int32)
                                               :node-size   (ml/linear 1 20 20 :int32)
                                               :sample-rate (ml/linear 0.1 1 10)}))))

;; ### Evaluate a single model
(defn evaluate-model [dataset split-fn model-type model-fn]
  (let [data-split (split-fn dataset)
        pipelines (map model-fn (generate-hyperparams model-type))]
    (evaluate-pipe pipelines data-split)))

;; ### Split functions
; In fact, having two splits is not necessary because I do not have a 2nd set of parameters to tune such as a cutoff value, for example. I could simply use the `train-test` function on my dataset to cross-validate a well tuned models according to my grid and the fits per hyperparameter.
(defn train-test [dataset]
  (ds/split->seq dataset :bootstrap {:seed 123 :repeats 25}))

(defn train-val [dataset]
  (let [ds-split (train-test dataset)]
    (ds/split->seq (:train (first ds-split)) :kfold {:seed 123 :k 5})))

;; ### Define model types and corresponding functions as a vector of vectors
(def model-type-fns
  {:gradient-tree gradient-tree-pipe-fn
   :random-forest random-forest-pipe-fn})

;; ### Evaluate models for a dataset
(defn evaluate-models [dataset split-fn]
  (mapv (fn [[model-type model-fn]]
          (evaluate-model dataset split-fn model-type model-fn))
        model-type-fns))

;; ### Evaluate separately
(def tree-models (evaluate-models concrete-data train-val))

;; ## Extract usable models
; This is a function in past assignments I stored away in `src/utils`. I might as well have it in the code as it's a work horse, restructuring the deeply nested maps of models created and evaluated.
(defn best-models [eval]
  (->> eval
       flatten
       (map
         #(hash-map :summary (ml/thaw-model (get-in % [:fit-ctx :model]))
                    :fit-ctx (:fit-ctx %)
                    :timing-fit (:timing-fit %)
                    :metric ((comp :metric :test-transform) %)
                    :other-metrices ((comp :other-metrices :test-transform) %)
                    :other-metric-1 ((comp :metric first) ((comp :other-metrices :test-transform) %))
                    :other-metric-2 ((comp :metric second) ((comp :other-metrices :test-transform) %))
                    :params ((comp :options :model :fit-ctx) %)
                    :pipe-fn (:pipe-fn %)))
       (sort-by :metric)))

;; ### Gradient tree
(def best-val-gradient-tree
  (-> (first tree-models)
      best-models
      reverse))

;; #### GT summary
(-> best-val-gradient-tree first :summary)
(-> best-val-gradient-tree first :metric)
(-> best-val-gradient-tree first :other-metrices)
(-> best-val-gradient-tree first :params)

; Just by observing the hyperparameters--viz. max-depth, max-nodes, and node-size--I trust that tree is over-fit. We will observe the learning curve when building the final model below.

^kind/smile-model
(-> best-val-gradient-tree first :summary .trees seq)

;; ### Random Forest
(def best-val-random-forest
  (-> (second tree-models)
      best-models
      reverse))

;; #### RF summary
(-> best-val-random-forest first :summary)
(-> best-val-random-forest first :metric)
(-> best-val-random-forest first :other-metrices)
(-> best-val-random-forest first :params)

; Same (potential) overfitting idea here.

;; ## Build final models for evaluation
;; ### Gradient tree
(def final-model-gradient-tree
  (-> (evaluate-pipe
        [(-> best-val-gradient-tree first :params
             gradient-tree-pipe-fn)]
        (train-test concrete-data))
      best-models))

;; #### GT FINAL
(-> final-model-gradient-tree first :summary)
(-> final-model-gradient-tree first :metric)
(-> final-model-gradient-tree first :other-metrices)
(-> final-model-gradient-tree first :params)

;; ### Learning curve
^kind/vega
(ml-viz/learnining-curve (-> concrete-data train-test first :train)
                         (-> best-val-gradient-tree first :params
                             gradient-tree-pipe-fn)
                         [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
                         {:k                5
                          :metric-fn        ml/mae
                          :loss-or-accuracy :loss}
                         {:YSCALE      {:zero false}
                          :TRAIN-COLOR "green"
                          :TEST-COLOR  "red"
                          :TITLE       "Learning Curve"
                          :YTITLE      "MAE"})

; Indeed, we find a wide gap at the beginning of the plot. Recall that a wide gap indicates higher variance. Eventually, by including 100% of the training data, the gap narrows.

(-> final-model-gradient-tree first :summary .trees seq)

;; ### Variable importance
(def importance-tree
  (let [important (-> final-model-gradient-tree first :summary .importance seq)
        regressor regressors
        mapped (zipmap regressor important)
        sorted (sort-by second mapped)]
    (reverse sorted)))

(def map-of-importance-tree
  (map (fn [[x y]] {:x x :y y}) importance-tree))

^kind/vega
(-> map-of-importance-tree
    (hanami/plot ht/bar-chart {:X :x :Y :y :XTYPE "nominal" :XSORT "-y"
                               :XTITLE "Variable" :YTITLE "Importance"
                               :TITLE "GT Variables of Importance"}))

;; ### Random forest
(def final-model-random-forest
  (-> (evaluate-pipe
        [(-> best-val-random-forest first :params
             random-forest-pipe-fn)]
        (train-test concrete-data))
      best-models))

(-> final-model-random-forest first :summary)
(-> final-model-random-forest first :metric)
(-> final-model-random-forest first :other-metrices)
(-> final-model-random-forest first :params)

;; ### Learning curve
^kind/vega
(ml-viz/learnining-curve (-> concrete-data train-test first :train)
                         (-> best-val-random-forest first :params
                             random-forest-pipe-fn)
                         (map #(/ % 10) (vec (range 1 11)))
                         {:k                5
                          :metric-fn        ml/mae
                          :loss-or-accuracy :loss}
                         {:YSCALE      {:zero false}
                          :TRAIN-COLOR "green"
                          :TEST-COLOR  "red"
                          :TITLE       "Learning Curve"
                          :YTITLE      "MAE"})

;; ### Variable importance
(def importance-forest
  (let [important (-> final-model-random-forest first :summary .importance seq)
        regressor regressors
        mapped (zipmap regressor important)
        sorted (sort-by second mapped)]
    (reverse sorted)))

(def map-of-importance-forest
  (map (fn [[x y]] {:x x :y y}) importance-forest))

^kind/vega
(-> map-of-importance-forest
    (hanami/plot ht/bar-chart {:X :x :Y :y :XTYPE "nominal" :XSORT "-y"
                               :XTITLE "Variable" :YTITLE "Importance"
                               :TITLE "RF Variables of Importance"}))
