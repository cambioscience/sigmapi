(ns sigmapi.test.core
  (:require
    [clojure.test :refer [deftest testing is]]
    [clojure.math :as maths :refer [pow exp PI sqrt log ceil floor round]]
    [sigmapi.core :as sp :refer :all]
    [clojure.core.matrix :as m]
    [kixi.stats.distribution :as xd]
    [kixi.stats.core :as xc]
    [com.hypirion.clj-xchart :as ch]
    [loom.graph :as lg]
    [loom.alg :as la]
    [loom.io :as lio]
    [emmy.env :as e]))

(defn e= [e x y] (< (Math/abs (- x y)) e))

(defn
  fg-test-graph-f7
  "Figure 7 in Frey2001 Factor graphs and the sum product algorithm"
  ([] (fg-test-graph-f7 :sp/sp))
  ([alg] (fg-test-graph-f7 alg (lg/graph ['fa 'x1] ['fb 'x2] ['x1 'fc] ['x2 'fc] ['fc 'x3] ['x3 'fd] ['x3 'fe] ['fd 'x4] ['fe 'x5])))
  ([alg g]
   (fg-test-graph-f7 alg g {'x5 #{0 1} 'x2 #{0 1 2} 'x3 #{0 1 2 3} 'x4 #{0 1} 'x1 #{0 1}}))
  ([alg g states-map]
   {
    :states   states-map
    :messages {}
    :graph    g
    :nodes    {
               'x1 (make-node {:alg alg :type :sp/variable :id 'x1})
               'x2 (make-node {:alg alg :type :sp/variable :id 'x2})
               'x3 (make-node {:alg alg :type :sp/variable :id 'x3})
               'x4 (make-node {:alg alg :type :sp/variable :id 'x4})
               'x5 (make-node {:alg alg :type :sp/variable :id 'x5})
               'fa (make-node {:alg alg :type :sp/factor :graph g :id 'fa :cpm (m/matrix [0.25 0.75]) :dfn {'x1 0}})
               'fb (make-node {:alg alg :type :sp/factor :graph g :id 'fb :cpm (m/matrix [0.19 0.9 0.452]) :dfn {'x2 0}})
               'fc (make-node {:alg alg :type :sp/factor :graph g :id 'fc :cpm (random-matrix [2 3 4]) :dfn {'x1 0 'x2 1 'x3 2}})
               'fd (make-node {:alg alg :type :sp/factor :graph g :id 'fd :cpm (random-matrix [4 2]) :dfn {'x3 0 'x4 1}})
               'fe (make-node {:alg alg :type :sp/factor :graph g :id 'fe :cpm (random-matrix [4 2]) :dfn {'x3 0 'x5 1}})
               }}))

(defn figure7
  "Figure 7 in Frey2001 Factor graphs and the sum product algorithm"
  ([]
   {
    :fg
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
    [
     s [2 3 4 5]
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

; combine-by-tranz [f g messages to dim-for-node]
(deftest test-combine-by-tranz
  (testing "That adding a sequence of vectors containing the value of their length
  to a matrix of the same shape as the sequence of vectors results in a matrix having
  every value equal to the sum of its dimensions"
    (is (m/equals (test-cbt) (m/fill (m/new-array [2 3 4 5]) (reduce + [2 3 4 5]))))))

(deftest test-max-configuration
  (testing "That a simple graph (a branch, x2->x1<-x3) returns max config"
    (->>
      (fgtree
        (:x1
          [:x1x2
           [
            [0.1 0.2 0.7]
            [0.6 0.2 0.2]
            ]
           (:x2 [0.2 0.8])]
          [:x1x3
           [
            [0.5 0.1 0.4]
            [0.8 0.1 0.1]
            ]
           (:x3 [0.3 0.6 0.1])]))
      exp->fg :sp/mxp
      propagate
      MAP-config)))

(defn t1 []
  (->>
      (fgtree
        (:x1
          [:x1x2
           [
            [0.1 0.2 0.7]
            [0.6 0.2 0.2]
            ]
           (:x2 [0.2 0.8])]
          [:x1x3
           [
            [0.5 0.1 0.4]
            [0.8 0.1 0.1]
            ]
           (:x3 [0.3 0.6 0.1])]))
       (exp->fg :sp/mxp)
       propagate
       MAP-config))

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
       {h :h} (-> experiment sp/updated-variables :marginals)
       expected [0.2463 0.3547 0.3990]
       result (map (fn [hv ev] [hv ev (e= 10e-5 hv ev)]) h expected)
     ]
     {
       :expected expected
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
      {:fg (fgtree
             (:host's-choice
               [:your-1st-choice|host's-choice
                [
                 [0 1/2 1/2]
                 [1/2 0 1/2]
                 [1/2 1/2 0]
                 ]
                (:your-1st-choice
                  [:prize|your-1st-choice&door
                   [
                    [[0 1] [1 0] [1 0]]
                    [[1 0] [0 1] [1 0]]
                    [[1 0] [1 0] [0 1]]
                    ]
                   (:door-0 [:p-door-0 [1/3 1/3 1/3]])
                   (:prize-0)]
                  [:p-your-1st-choice [1/3 1/3 1/3]])]
               [:door|host's-choice
                [
                 [0 1/2 1/2]
                 [1/2 0 1/2]
                 [1/2 1/2 0]
                 ]
                (:door [:p-door [1/3 1/3 1/3]])]
               [:your-2nd-choice|host's-choice
                [
                 [0 1/2 1/2]
                 [1/2 0 1/2]
                 [1/2 1/2 0]
                 ]
                (:your-2nd-choice
                  [:your-2nd-choice|your-1st-choice
                   [
                    [0 1/2 1/2]
                    [1/2 0 1/2]
                    [1/2 1/2 0]
                    ]
                   (:your-1st-choice-0 [:p-your-1st-choice-0 [1/3 1/3 1/3]])]
                  [:prize|your-2nd-choice&door
                   [
                    [[0 1] [1 0] [1 0]]
                    [[1 0] [0 1] [1 0]]
                    [[1 0] [1 0] [0 1]]
                    ]
                   (:door-1 [:p-door-1 [1/3 1/3 1/3]])
                   (:prize-1)])]))
       :priors
       {:door :p-door
        :door-0 :p-door-0
        :door-1 :p-door-1
        :your-1st-choice :p-your-1st-choice
        :your-1st-choice-0 :p-your-1st-choice-0}}
      door (or dp (assoc [0 0 0] door-number 1))
      choice (or cp (assoc [0 0 0] choose-door-number 1))
      {m1 :marginals l :updated :as em0}
      (-> model (assoc :data {:p-door door :p-door-0 door :p-door-1 door :p-your-1st-choice choice :p-your-1st-choice-0 choice}) sp/update-priors)
      m2
      (-> l (assoc :alg :sp/mxp) sp/change-alg propagate MAP-config)
      ]

     {:result (if (== 1 (:prize-1 m2)) '🚗 '🐐)
      :model l}
     )))



(defn normal
  [scale mean sd]
  (fn [x]
    (* scale
      (exp
        (* -1
          (/ (pow (- x mean) 2)
             (* 2 (pow sd 2))))))))


(comment


  (:result (MHP {:correct-door (rand-int 3) :choose-door (rand-int 3)}))

  (frequencies
    (repeatedly 100
      (fn [] (:result (MHP {:correct-door (rand-int 3) :choose-door (rand-int 3)})))))




  (->>
      (fgtree
        (:t
          [:r|t
           [
            [1/2 1/2 0]
            [0 2/3 1/3]
            [1/3 1/3 1/3]
            ]
           (:r [:pr [1/6 2/3 1/6]])]
          [:pt [1/3 1/3 1/3]]))
     (exp->fg :sp/mxp)
    (propagate (comp (fn [m] (println (update-vals (:messages m) keys)) m) message-passing))
    :end
    ;:messages
    ;MAP-config
    )


  ; (t) [r|t []]
  ; [r|t] (r)
  ; r|t r

  (let [
        r|c {:id :r|c, :matrix [[0.9 0 0.1] [1/6 1/6 2/3] [1/3 1/3 1/3]]}
        r|t {:id :r|t, :matrix [[1/3 2/3 0] [0 1/6 5/6] [1/3 1/3 1/3]]}
        ]
    (->>
     (edges->fg :sp/mxp
       [[{:id :t} r|t]
        [r|t {:id :r}]
        [{:id :r} {:id :pr, :matrix [1/6 1/6 2/3]}]
        [{:id :t} {:id :pt, :matrix [1/3 1/3 1/3]}]

        [{:id :c} r|c]
        [r|c {:id :r}]
        [{:id :c} {:id :pc :matrix [2/3 1/6 1/6]}]

        [{:id :t|c, :matrix [[0.1 0 0.9] [1/6 2/3 1/6] [1/3 1/3 1/3]]} {:id :t}]

        ])
     propagate
     MAP-config))


  (let [
         s|p&b {:id :s|p&b, :matrix
                [
                 [[1 0] [1 0]]
                 [[1 0] [0 1]]
                 ]}
        ]
    (->>
     (edges->fg :sp/mxp
       [
        [{:id :p} {:id :pp, :matrix [1/2 1/2]}]
        [{:id :b} {:id :pb, :matrix [1/2 1/2]}]
        [{:id :s} {:id :ps, :matrix [1 0]}]

        [{:id :p} s|p&b]
        [{:id :b} s|p&b]
        [s|p&b {:id :s}]

        ])
     propagate
     MAP-config))


  (let [
         w|s&r {:id :w|s&r, :matrix
                [
                 [[1 0] [0 1]]
                 [[0 1] [0 1]]
                 ]}
         s|y {:id :s|y :matrix [[0.9 0.1] [1/3 2/3]]}
         r|y {:id :r|y :matrix [[1/3 2/3] [2/3 1/3]]}
        ]
    (->>
     (edges->fg :sp/sp
       [
        [{:id :y} {:id :py, :matrix [1/2 1/2]}]
        [{:id :s} {:id :ps, :matrix [0 1]}]
        [{:id :r} {:id :pr, :matrix [1/2 1/2]}]
        [{:id :w} {:id :pw, :matrix [1/2 1/2]}]

        [{:id :y} s|y]
        [{:id :y} r|y]
        [s|y {:id :s}]
        [r|y {:id :r}]
        [{:id :s} w|s&r]
        [{:id :r} w|s&r]
        [w|s&r {:id :w}]

        ])
      (propagate-cycles 8)

      ;last
      ;doall
      ;:end
      ;(map (comp println print-msgs))
      ;doall
      (map (comp normalize-vals unnormalized-marginals))
      ;last
      ;marginals
      ;normalize-vals
      ))


  (->>
    '{:edges
      [
       z    y|z
       y|z  y

       z    x|z
       x|z  x

       x|   x
       z|   z
       ]
      :nodes
        {
          y|z
           [[0.8 0.1]
            [0.1 0.8]]
          x|z
           [[1/2 1/2]
            [1/2 1/2]
            ]
           x| [0 1]
           z| [1 0]
         }}
    (graph->fg :sp/sp)
    ;:graph
    ;((fn [g] (lio/view g {:alg :neato :node-label name })))
    propagate
    unnormalized-marginals
    normalize-vals
    ;MAP-config
    )


  (->>
    '{:edges
      [
       z    y|x&z
       x    y|x&z

       y|x&z  y

       z    x|z
       x|z  x


       x|   x
       z|   z
       ;y|   y
       ]
      :nodes
        {
          y|x&z
           [
             [[0.07 0.93] [0.13 0.87]]
             [[0.27 0.73] [0.31 0.69]]
           ]
          x|z
           [
            [0.3 0.7]
            [0.8 0.2]
            ]
          x| [0 1]                                                          ; drug
          z| [0.5 0.5]                                                             ; gender
         ;y| [1 0]                                                                 ; recovery
         }}
    (graph->fg :sp/sp)
    ;:graph
    ;((fn [g] (lio/view g {:alg :neato :node-label name })))
    (propagate-cycles 9)
    last
    marginals
    named-marginals
    )

  (let [model
        '{:edges
          [
           z    y|x&z
           x    y|x&z
           y|x&z  y

           x|   x
           z|   z
           ]
          :nodes
            {
              y|x&z
               [
                 [[0.07 0.93] [0.13 0.87]]
                 [[0.27 0.73] [0.31 0.69]]
               ]
              x| [0 1]
              z| [0.5 0.5]
             ;y| [1 0]
             }
          :states {:x [:drug :no-drug] :y [:didn't-recover :recovered] :z [:male :female]}
          :aliases {:x :drug :y :recovery :z :gender}}]
      (->> model
       (graph->fg :sp/sp)
       ;:graph
       ;((fn [g] (lio/view g {:alg :neato :node-label name })))
       (propagate-cycles 2)
       last
       marginals
       (named-marginals model)
       ))


  (defn intervene [model variable amount]
    (let [v (get-in model [:nodes variable])
          s (m/shape v)
          d (last s)
          p (/ 1 d)]
      (assoc-in model [:nodes variable]
        (m/emap (fn [x] (+ (* amount p) (* (- 1 amount) p x))) v))))

  (let [model
          '{:edges
            [
             z    y|z&x
             x    y|z&x
             y|z&x  y

             z    x|z
             x|z  x

             x|   x ;
             z|   z ;
             ;y|   y
             ]
            :nodes
              {
                y|z&x
                 [
                   [[0.07 0.93] [0.13 0.87]]
                   [[0.27 0.73] [0.31 0.69]]
                 ]
                x|z
                 [
                   [0.3 0.7]
                   [0.8 0.2]
                 ]
                x| [1 0]
                z| [0.5 0.5]
               ;y| [1 0]

               }
            :states {:x [:drug :no-drug] :y [:didn't-recover :recovered] :z [:male :female]}
            :aliases {:x :drug :y :recovery :z :gender}}
        intervened-model
          (assoc-in model [:nodes 'x|z]
            [
              [0.5 0.5]
              [0.5 0.5]
            ])
        ;im (intervene model 'x|z 0.5)
        ]
    (->> intervened-model
     (graph->fg :sp/sp)
     ;:graph
     ;((fn [g] (lio/view g {:alg :neato :node-label name })))
     (propagate-cycles 8)
     last
     marginals
      (named-marginals model)
    ))

(let [model
        '{:edges
          [
           ;g s|g
            s|g s
            s t|s
            t|s t
            t c|g&t
            g c|g&t
            c|g&t c
            g| g
            s| s
            t| t
            c| c
          ]
          :nodes
          {
           c|g&t
           [
            [[0.5 0.5] [0.45 0.55]]
            [[0.2 0.8] [0.2 0.8]]
            ]
           s|g
           [
            [0.5 0.5]
            [0.2 0.8]
            ]
           t|s
           [
            [0.5 0.5]
            [0.5 0.5]
            ]
           s| [0 1]
           t| [0.5 0.5]
           g| [0.5 0.5]
           c| [0.5 0.5]
           }
          :states {:s [:not-smoking :smoking] :c [:no-cancer :cancer] :t [:no-tar :tar] :g [:put-gene-0 :put-gene-1]}
          :aliases {:s :smoking :c :cancer :t :tar :g :genotype}}
        intervened-model
        (assoc-in model [:nodes 't|s]
          [
           [0.5 0.5]
           [0.5 0.5]
           ])
        ;im (intervene model 'x|z 0.5)
        ]
    (->> model
     (graph->fg :sp/sp)
      ;:graph
      ;((fn [g] (lio/view g {:alg :neato :node-label name })))
      (propagate-cycles 8)
      last
      marginals
      (named-marginals model)
    ))


(let [model
          '{:edges
            [
             z    y|z&x
             x    y|z&x
             y|z&x  y

             z    x|z
             x|z  x

             x|   x ;
             z|   z ;
             ;y|   y
             ]
            :nodes
              {
                y|z&x
                 [
                   [[0.07 0.93] [0.13 0.87]]
                   [[0.27 0.73] [0.31 0.69]]
                 ]
                x|z
                 [
                   [0.3 0.7]
                   [0.8 0.2]
                 ]
                x| [1 0]
                z| [0.5 0.5]
               ;y| [1 0]
               }
            :states  {:x [:a :b] :y [:thin :fat] :z [:thin :fat]}
            :aliases {:x :diet :y :final-weight :z :initial-weight}}
        intervened-model
          (assoc-in model [:nodes 'x|z]
            [
              [0.5 0.5]
              [0.5 0.5]
            ])
        ]
    (->> model
     (graph->fg :sp/sp)
     ;:graph
     ;((fn [g] (lio/view g {:alg :neato :node-label name })))
     (propagate-cycles 8)
     last
     marginals
     (named-marginals model)
    ))

(cc/quick-bench (MHP))

 (m/set-current-implementation :vectorz)

  (let [mat (m/matrix (m/reshape (range 27) [3 3 3]))]
    (type (first (tranz mat 2 first))))

  (require '[criterium.core :as cc])

  (let [mat (-> (m/reshape (range 27) [3 3 3]) m/matrix)]
    (cc/quick-bench (-> mat (m/transpose [2 1 0]) )))








































  "

  Definition 3.3.1 (The Backdoor Criterion)

  Given an ordered pair of variables (X,Y) in a directed acyclic graph G,
  a set of variables Z satisfies the backdoor criterion relative to (X, Y)
  if no node in Z is a descendant of X,
  and Z blocks every path between X and Y that contains an arrow into X.

  and nodes in Z are observed



  condition on paths



  "


  "
  blocked by set Z:

  unconditional: colliders

  conditional (conditioned nodes Z):
    * unconditioned colliders not in Z and having no descendants in Z
    * chain or fork whose middle node is in Z

  blocking is conditioning

  "















 (ch/view
   (ch/xy-chart
     {"os" (ch/extract-series
             {:x first
              :y last}
             (reduce
               (fn [[x u y] [x' u' y']]
                 [x (+  )])
               (map
                (fn [x u]
                  [x u (+ x u)]) (range 0 1 0.1) (repeatedly (fn [] (* 0.1 (rand)))))))}
     {:title "4096 random nodes from all spaces - best LDA score, nc/nn"
      :y-axis {:title "nl" :decimal-pattern "##.####"}
      :x-axis {:title "nc" :decimal-pattern "##.####"}
      :render-style :scatter}))

  (require '[clojure.java.io :as io])
  (require '[clojure.string :as string])








  [[1.0 0.5 0.3]
   [0.1 1.0 0.8]
   [0.2 0.4 1.0]
   ]



  (lio/view (apply lg/weighted-digraph
              (remove (comp zero? last) (apply concat
                 (map-indexed
                   (fn [i r] (map-indexed (fn [j x] [i j x]) r))
                   (partition 3
                     [
                      0 0.5 1
                      0 0 1
                      0 0 0
                      ]))))))






(reduce + (map (fn [x] (* 0.5 (exp (- (pow x 2))))) (range -3 3 0.5)))

(sqrt PI)

(apply max-key identity (concat (range 0 0.9 0.1) (range 0.9 0 -0.1)))

(apply max-key log (concat (range 0 0.9 0.1) (range 0.9 0 -0.1)))

(map log (concat (range 0 0.9 0.1) (range 0.9 0 -0.1)))


 (xd/normal {:mu 0 :sd 1})


  (defn normal [mu sd]
    (fn [x] (* (/ 1 (* sd (sqrt (* 2 PI)))) (exp (* -1/2 (pow (/ (- x mu) sd) 2))))))

  (* 0.1 (reduce + (map (normal 0 1) (range -10 10 0.1))))

  (let [r (range -5 5 0.1)
        a (map (normal -1 0.2) r)
        b (map (normal 2 0.7) r)
        c (let [ab (map * a b) sab (/ 1 (reduce + (map (partial * 0.1) ab)))] (map (partial * sab) ab))]
    (ch/view
     (ch/xy-chart
       {
        "a" {:x r :y a}
        "b" {:x r :y b}
        "c" {:x r :y c}
        }
       {})))


  (count (range -5 5 0.1))

  (test-Bayesian-updating)


  (let
    [model
     {:fg
      (sp/fgtree
        (:d [:pd [0.5 0.5]]
          [:h|d&e
           [
            [[0.6 0.5 0.4] [0.8 0.5 0.2]]
            [[0.4 0.5 0.6] [0.2 0.5 0.8]]
            ]
           (:e [:pe [1/2 1/2]])
           (:h [:ph [1/3 1/3 1/3]]
             [:pd|h
              [
               [0.9 0.1]
               [0.5 0.5]
               [0.1 0.9]
               ]
              (:dp)
              ])
           ]))
      :priors
      {:h :ph :d :pd :e :pe}}
     ]
    (->>
      (reductions
        (fn [{{p :dp} :marginals :as m} {d :pd :as data}]
          (let [e (min 1 (max 0 (int (ceil (dec (reduce + (map (comp abs -) d p)))))))
                pe (assoc [0 0] e 1)]
            (update-priors
             (assoc m :data (assoc data :pe pe)))))
        model
        (->> [{:pd [1/2 1/2]}]
          (into (interleave (repeat 8 {:pd [1 0]}) (repeat 8 {:pd [0 1]})))
          (into (repeat 8 {:pd [0 1]}))))
      (map (juxt :marginals :data))
      ))


  ((e/D e/square) 'x)

  (e/D (e/square 'x))

  (e/square 'x)

  (map (fn [x] ((e/log (normal 1 0 1)) x)) (range -2 2 0.1))

  ((e/log (normal 1 0 1)) x)

  (e/simplify (e/log (normal 1 0 1)))

  (e/simplify
    (e/log
      (e/*
        (e// 1 (e/* 'sd (e/sqrt (e/* 2 PI))))
        (e/exp (e/* -1/2 (e/expt (e// (e/- 'x 'mu) 'sd) 2))))))


  '(/ (+ (* -1/2 (expt mean 2)) (* mean x) (* -1/2 (expt x 2)))
      (expt sd 2))

 (apply e/+ )



  )