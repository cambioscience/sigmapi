(ns sigmapi.test.core
  (:require
    [clojure.stacktrace :refer [print-cause-trace]]
    [clojure.test :as test :refer [deftest testing is]]
    [clojure.math :as maths :refer [pow exp PI sqrt log ceil floor round]]
    [sigmapi.core :as sp :refer :all]
    [sigmapi.impl.core-matrix :as spt :refer [random-matrix combine P]]
    [sigmapi.impl.normal :as spn :refer [multivariate-normal normal]]
    [clojure.core.matrix :as m]
    [loom.graph :as lg]
    [loom.alg :as la]
    [loom.io :as lio]
    [criterium.core :as cc]))



(defn e= [e x y] (< (Math/abs (- x y)) e))

(defn figure7
  "Figure 7 in Frey2001 Factor graphs and the sum product algorithm"
  ([]
   {:fg
    (fgtree
      [:fc
       [
        [[0.3 0.3 0.3 0.1] [0.3 0.3 0.3 0.1] [0.3 0.3 0.3 0.1]]
        [[0.3 0.3 0.3 0.1] [0.3 0.3 0.3 0.1] [0.3 0.3 0.3 0.1]]
        ]
       (:x1 [:fa [0.1 0.9]])
       (:x2 [:fb [0.2 0.7 0.1]])
       (:x3
         [:fd
          [
           [0.4 0.6]
           [0.6 0.4]
           [0.4 0.6]
           [0.6 0.4]
           ]
          (:x4)]
         [:fe
          [
           [0.5 0.5]
           [0.5 0.5]
           [0.5 0.5]
           [0.5 0.5]
           ]
          (:x5)])])
    :priors {:x1 :fa :x2 :fb}
    }))

(defn test-cbt []
  (let
    [s [2 3 4 5]
     f (m/new-array s)
     g (fn ([mat v] (m/add mat v)) ([mat] mat))
     vs (map (fn [d] (m/matrix (repeat d d))) s)
     dfn (into {} (map vector s (range (count s))))
     to 2
     msgs (map (fn [v i] {:value v :id i}) vs s)
     ; do them out-of-order in as messages may not come in dimension order
     reordered-msgs (mapcat (fn [[a b]] [b a]) (partition-all 2 msgs))
     sum (combine f g reordered-msgs to dfn)
     ]
    sum))

(deftest test-combine-by-tranz
  (testing "That adding a sequence of vectors containing the value of their length
  to a matrix of the same shape as the sequence of vectors results in a matrix having
  every value equal to the sum of its dimensions"
    (is (m/equals (test-cbt) (m/fill (m/new-array [2 3 4 5]) (reduce + [2 3 4 5]))))))

(defn max-config-test []
  (->>
      (fgtree
        (:x1
          [:x2|x1
           [
            [0.1 0.2 0.7]
            [0.6 0.2 0.2]
            ]
           (:x2 [0.2 0.8])]
          [:x3|x1
           [
            [0.5 0.1 0.4]
            [0.8 0.1 0.1]
            ]
           (:x3 [0.3 0.6 0.1])]))
      (exp->fg :MAP :core.matrix/tensor)
      propagate
      MAP-config))

(deftest test-max-configuration
  (testing "That a simple graph (a branch, x2<-x1->x3) returns max config"
    (is (= (max-config-test) {:x1 1, :x3 0, :x2 0}))))

(defn test-Bayesian-updating
  "
    An example from:
    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading11.pdf
    part 4 Updating again and again
  "
  []
  (let
    [model
     {:fg
      (sp/fgtree
        (:d [:pd [0.5 0.5]]
          [:h|d
           [
            [0.5 0.4 0.1]
            [0.5 0.6 0.9]
            ]
           (:h [:ph [0.4 0.4 0.2]])
           ]))
      :priors
      {:h :ph :d :pd}}
     experiment
       (assoc model :data
         [
          {:pd [0 1]}
          {:pd [0 1]}
          ])
       {h :h} (-> experiment spt/updated-variables :marginals)
       expected [0.2463 0.3547 0.3990]
       result (map (fn [hv ev] [hv ev (e= 10e-5 hv ev)]) h expected)
     ]
     {:expected expected
      :result h
      :pass? (every? true? (map last result))
      :experiment experiment
     }))

(defn MHP
  "
     Suppose you're on a game show,
     and you're given the choice of three doors:
     Behind one door is a car;
     behind the others, goats.
     You pick a door, say No. 1,
     and the host, who knows what's behind the doors,
     opens another door, say No. 3, which has a goat.
     He then says to you,
     'Do you want to pick door No. 2?'
     Is it to your advantage to switch your choice ?
  "
  ([]
   (MHP {}))
  ([{door-number :correct-door choose-door-number :choose-door dp :dp cp :cp
    :or {door-number 1 choose-door-number 0}}]
   (let
     [model
      {:fg
       (fgtree
         (:door [:p-door [1/3 1/3 1/3]]
           [:your-1st-choice|door
            [
             [1/3 1/3 1/3]
             [1/3 1/3 1/3]
             [1/3 1/3 1/3]
             ]
            (:your-1st-choice [:p-your-1st-choice [1/3 1/3 1/3]])]
           [:host's-choice|door
            [
             [0 1/2 1/2]
             [1/2 0 1/2]
             [1/2 1/2 0]
             ]
            (:host's-choice
              [:your-2nd-choice|host's-choice
                [
                 [0 1/2 1/2]
                 [1/2 0 1/2]
                 [1/2 1/2 0]
                 ]
                (:your-2nd-choice
                  [:prize|your-2nd-choice&door
                   [
                    [[0 1] [1 0] [1 0]]
                    [[1 0] [0 1] [1 0]]
                    [[1 0] [1 0] [0 1]]
                    ]
                   (:door' [:p-door' [1/3 1/3 1/3]])
                   (:prize)])])]))
       :priors
       {:door :p-door
        :door' :p-door'
        :your-1st-choice :p-your-1st-choice}}
      door (or dp (assoc [0 0 0] door-number 1))
      choice (or cp (assoc [0 0 0] choose-door-number 1))
      {m1 :marginals l :updated :as em0}
      (-> model
        (assoc
          :impl :core.matrix/tensor
          :data {:p-door door :p-door' door :p-your-1st-choice choice})
        spt/update-priors)
      m2
      (-> l (assoc :alg :MAP :impl :core.matrix/tensor) sp/change-alg propagate MAP-config)
      ]
     {:result (if (== 1 (:prize m2)) '🚗 '🐐)
      :model l :config m2 :marginals m1})))


(comment


  (into (sorted-map)
    (frequencies
     (repeatedly 100
       (fn [] (:result (MHP {:correct-door (rand-int 3) :choose-door (rand-int 3)}))))))



  (test-Bayesian-updating)

  (max-config-test)

  (test/run-tests)


)
