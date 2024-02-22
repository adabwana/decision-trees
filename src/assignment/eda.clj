(ns assignment.eda
  (:require
    [clojure.math.combinatorics :as combo]
    [fastmath.stats :as stats]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.ml.dataset :as ds]))

;; # Exploratory Data Analysis
;; Load data
(defonce concrete-data
         (ds/dataset "data/Concrete_Data.csv"
                     {:key-fn (fn [colname]
                                (-> colname
                                    (clojure.string/replace #"\.|\s" "-")
                                    clojure.string/lower-case
                                    (clojure.string/split #"\-\(|\--\(|\(")
                                    first
                                    keyword))}))

(ds/info concrete-data)

(def response :concrete-compressive-strength)
(def regressors
  (ds/column-names concrete-data (complement #{response})))

^kind/vega
(let [data (ds/rows concrete-data :as-maps)
      column-names (ds/column-names concrete-data)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x {:field {:repeat "column"} :type "quantitative"}
                       :y {:aggregate "count"}}}})

^kind/vega
(let [data (ds/rows concrete-data :as-maps)
      column-names (ds/column-names concrete-data)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:width    60 :mark "boxplot"
            :encoding {:y {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}}}})

(let [columns (ds/column-names concrete-data)]
  (for [column columns]
    (vector column (count (stats/outliers (get concrete-data column))))))

;; ### Pairs-plots.
^kind/vega
(let [data (ds/rows concrete-data :as-maps)
      column-names (ds/column-names concrete-data)]
  {:data   {:values data}
   :repeat {:column column-names
            :row    column-names}
   :spec   {:height   100 :width 100
            :mark     "circle"
            :encoding {:x {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}
                       :y {:field {:repeat "row"} :type "quantitative" :scale {:zero false}}}}})

(let [combos (combo/combinations regressors 2)]
  (for [[x y] combos]
    (assoc {} [x y] (stats/correlation (get concrete-data x) (get concrete-data y)))))

(for [[x y] (mapv (fn [r] [response r]) regressors)]
  (assoc {} [x y] (stats/correlation (get concrete-data x) (get concrete-data y))))