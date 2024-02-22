(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []                                              ;for clay 63
  (clay/make!
    {:format              [:quarto :html]
     :book                {:title "Decision Trees, Boosting, Bagging"}
     :base-source-path    "src"
     :base-target-path    "docs"                            ;default
     :subdirs-to-sync     ["notebooks" "data"]
     :clean-up-target-dir true
     :source-path         ["index.clj"                      ;index.md
                           "assignment/eda.clj"
                           "assignment/clojure_smile.clj"
                           "assignment/r_caret.clj"]}))

(defn build-book []                                         ;for clay >63
  (clay/make!
    {:base-source-path    "src"
     :base-target-path    "docs"                            ;default
     :title               "Linear Discriminate Analysis"
     ;:page-config ;configured in clay.edn
     :subdirs-to-sync     ["notebooks" "data"]
     :clean-up-target-dir true
     :quarto-book-config
     {:format               [:quarto :html]
      :chapter-source-paths ["index.clj"                    ;index.md
                             "assignment/eda.clj"
                             "assignment/decision_tree.clj"
                             "assignment/random_forest.clj"]}}))


(comment
  ;with index.md clay wont find in src and complains about docs/_book
  (build)
  (build-book))