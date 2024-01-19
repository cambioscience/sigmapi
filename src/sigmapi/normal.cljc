(ns sigmapi.normal
  (:require
    [clojure.stacktrace :refer [print-cause-trace]]
    [clojure.math :as maths :refer [PI]]
    [clojure.core.matrix :as m]
    [kixi.stats.distribution :as xd]
    [emmy.env :as e :refer :all]
    [emmy.matrix :as em]
    [sigmapi.core :as sp :refer :all]))

(def shape (juxt em/num-rows em/num-cols))

(defn broadcast [f]
  (fn [[x & xs :as tx]]
    (f x)))

(defn normal [mu sd]
  (with-meta
    (fn [x]
     (*
       (/ 1 (* sd (sqrt (* 2 PI))))
       (exp (* -1/2 (expt (/ (- x mu) sd) 2)))))
    {:mu mu :sigma sd :sigma-1 (if (== 0 sd) 0 (/ 1 sd))}))

(defn normal-log [mu sd]
  (fn [x]
    (/
      (+
        (* -1 (expt sd 2) (log sd))
        (* -1/2 (expt mu 2))
        (* mu x)
        (* -0.9189385332046727 (expt sd 2))
        (* -1/2 (expt x 2)))
       (expt sd 2))))

(defn multivariate-normal-log [mu sigma]
  (fn [x]
    (let [sigma (apply em/by-rows sigma)
          mu (apply em/row mu)
          k (count x)
          x (apply em/row x)
          x-mu (- x mu)
          sigma-1 (em/invert sigma)
          det-sigma (em/determinant sigma)]
      (get-in
        (+
          (* -1/2 (em/transpose x-mu) sigma-1 x-mu)
          (* -0.9189385332046727 k)
          (* -1/2 (log det-sigma)))
        [0 0]))))

(defn multivariate-normal
  ([mu sigma]
   (let [mu (apply em/row mu)
         sigma (apply em/by-rows sigma)
         sigma-1 (em/invert sigma)]
     (multivariate-normal mu sigma sigma-1)))
  ([mu sigma sigma-1]
     (with-meta
       (fn [x]
         (let [k (count x)
               x (apply em/column x)
               x-mu (- x mu)
               det-sigma (em/determinant sigma)]
           (/
             (exp (get-in (* -1/2 (em/transpose x-mu) sigma-1 x-mu) [0 0]))
             (sqrt (* (expt (* 2 PI) k) det-sigma)))))
       {:mu mu :sigma sigma :sigma-1 sigma-1})))

(defn product-of-normals
  ([{:keys [f id dim-for-node messages to-dim zero-vector zero-matrix]}]
   (let
     [s1s (map (fn [{v :value id :id}] (let [j (dim-for-node id)] (assoc-in zero-matrix [j j] (:sigma-1 (meta v))))) messages)
      mus (map (fn [{v :value id :id}] (let [j (dim-for-node id)] (assoc-in zero-vector [0 j] (:mu (meta v))))) messages)
      sigma (em/invert (apply + (cons (:sigma-1 (meta f)) s1s)))
      mu (* (apply + (map * (cons (:mu (meta f)) mus) (cons (:sigma-1 (meta f)) s1s))) sigma)]
     {:mu mu :sigma sigma})))

(defn summarize
  "
    summing (integrating) over all variables except to
    is the same as the marginal of to (all the other variables are marginalized out)
    https://statproofbook.github.io/P/mvn-marg.html
  "
  ([mu sigma to-dim]
    (normal (em/get-in mu [0 to-dim]) (em/get-in sigma [to-dim to-dim]))))

(defmethod make-node [:sp :variable :normal]
  ([{:keys [id] :as node}]
   (with-meta node
     {; Messaging
     `><
      (fn [this messages to]
       (let [s1s (map (comp :sigma-1 meta :value) messages)
             mus (map (comp :mu meta :value) messages)
             s1+ (apply + s1s)
             sigma (if (== 0 s1+) s1+ (/ 1 s1+))
             mu (* sigma (apply + (map * s1s mus)))
             ]
         {
          :value (normal mu sigma)
          :repr (if (== 1 (count messages)) (:repr (first messages)) (cons '∏ (map :repr messages)))
          }))
     `<> (fn [this messages to to-msg parent-msg] (>< this messages to))
     `i
      (fn [this]
       {:value (with-meta (fn identity [x] (em/by-rows [1])) {:mu 0 :sigma 1 :sigma-1 1}) :repr id})})))

(defmethod make-node [:sp :factor :normal]
  ([{:keys [graph id clm cpm dfn] :as params}]
   (let [f (apply (if (number? (first cpm)) normal multivariate-normal) cpm)
         s1 (:sigma-1 (meta f))
         d  (if (number? s1) 0 (em/num-cols s1))
         zv (em/make-zero 1 d)
         zm (em/make-zero d)
         node
         (with-meta {:f f :id id :dim-for-node dfn :kind :factor :features #{:passes}
                     :zero-vector zv :zero-matrix zm :d d}
           { ; sp/Messaging
            `><
            (fn [{:keys [f id dim-for-node zero-vector zero-matrix d] :as this} messages to]
              (let [to-dim (dim-for-node to)
                    {:keys [mu sigma]} (product-of-normals (assoc this :to-dim to-dim :messages messages))
                    ]
                {
                 :value (summarize mu sigma to-dim)
                 :repr (cons '∑ (list (cons '∏ (list (:repr (i this)) (if (== 1 (count messages)) (:repr (first messages)) (map :repr messages))))))
                 }))
            `<> (fn [this messages to to-msg parent-msg] (>< this messages to))
            `i (fn [{:keys [f id dim-for-node zero-vector zero-matrix d]}]
                 {:value f :repr id :dim-for-node dim-for-node})
            ; Updatable
            `updated
              (fn [this {:keys [cpm]}]
                (assoc this :f (apply (if (number? (first cpm)) normal multivariate-normal) cpm)))
            })] node)))

(comment
  (deftype MaxNormalFactorNode
    [f id dim-for-node]
    Messaging
    (>< [this messages to]
      (let [
            to-dim (dim-for-node to)
            {:keys [mu sigma]} (product-of-normals f id dim-for-node messages to-dim)
            rsum (combine f m/add messages to dim-for-node)
            mm (map m/emin rsum)
            ]
        {
         :dim-for-node dim-for-node
         :value mm
         :min (indexed-min mm)
         :sum rsum
         :im (mapv (fn [[s c]] [s (zipmap (keys (dissoc dim-for-node to)) c)]) (map indexed-min rsum))
         :repr (list 'min (cons '∑ (cons (:repr (i this)) (map :repr messages))))
         }))
    (<> [this messages to to-msg parent-msg]
      (let [
            conf (get-in parent-msg [:configuration id])
            mind (zipmap (map :id messages) (range (count messages)))
            to-conf (get conf to)
            ]
       {
        :dim-for-node dim-for-node
        :value 0
        :mind mind
        :conf conf
        :configuration (assoc (:configuration parent-msg) to to-conf)
        }))
   (i [this] {:value f :repr id :dim-for-node dim-for-node})
   LogSpace
   (p [this x] (m/emap P x)))

 (deftype MaxNormalVariableNode
   [id]
   Messaging
   (>< [this messages to]
     (let [sum (apply m/add (map :value messages))]
       {
        :value sum
        :repr (cons '∑ (map :repr messages))
        }))
   (<> [this messages to to-msg parent-msg]
     (let
       [
        ; to-msg is the msg received by this node from to on the >< pass,
        ; which contains the indices of the other variables for each of this variable's states.
        ; Here we are telling to its configuration and the configurations of all previous variables
        ; In the outflowing messaging, the root variable node uses all its messages
        sum (apply m/add (map :value (cons to-msg messages)))
        min (indexed-min sum)
        ; look up the configuration we got in the forward pass which lead to this minimum
        ; (for the root - others need to use the indices they got from the parent)
        conf (if parent-msg (get-in parent-msg [:configuration id]) (get-in min [1 0]))
        configuration (if parent-msg (:configuration parent-msg) {id conf})
        mto (get-in to-msg [:im conf 1])
        ]
       {
        :value sum
        :min min
        :configuration (assoc configuration to mto)
        :repr (cons '∑ (map :repr messages))
        }))
   (i [this] {:value 0 :repr 0})
   Variable
   LogSpace
   (p [this x] (m/emap P x))))



(comment




(simplify
  (log
    (*
      (/ 1 (* 'sd (sqrt (* 2 PI))))
      (exp (* -1/2 (expt (/ (- 'x 'mu) 'sd) 2))))))


'(/
   (+
    (* -1 (expt sd 2) (log sd))
    (* -1/2 (expt mu 2))
    (* mu x)
    (* -0.9189385332046727 (expt sd 2))
    (* -1/2 (expt x 2)))
   (expt sd 2))


  (let [f (fn [mu sd]
            (fn [x]
               (/
                 (+
                  (* -1 (expt sd 2) (log sd))
                  (* -1/2 (expt mu 2))
                  (* mu x)
                  (* -0.9189385332046727 (expt sd 2))
                  (* -1/2 (expt x 2)))
                 (expt sd 2))))
        f+ (apply + (map :v [{:v (f 0 1)} {:v (f 0 1)}]))
        ]
    (f+ 0.5))



  (m/mmul [0.3 0.7] [[1 0.5] [0.5 1]] [0.9 0.4])

  (simplify
    (log
      (/
        (exp (* -1/2 'x-muT 'sigma-1 'x-mu))
        (sqrt (* (expt (* 2 PI) 'k) 'det-sigma)))))

(map (comp + log) (range 7))

  ((multivariate-normal-log [0 0] [[1 0.5] [0.5 1]]) [0 0])


  ((*
     (multivariate-normal [0 0] [[1 0.5] [0.5 1]])
     (broadcast (normal 0 1))) [2 2])

(sum
  (*
   (multivariate-normal [0 0] [[1 0.5] [0.5 1]])
   (fn identity [x] 1)) -1 1)


  (apply + [(apply em/by-rows [[1 0.5] [0.5 1]]) (apply em/by-rows [[1 0.5] [0.5 1]])])

  ((multivariate-normal [0 0] [[1 0.5] [0.5 1]]) [0 0])

  (* (expt (apply em/column [1 0.3]) 2) (apply em/by-rows [[1 0.5] [0.5 1]]))

  ((compose (literal-function 'f) (literal-function 'g)) 'x)

((broadcast (normal 0 1)) [0 0])

  (exp (em/column [2 3]))

  (exp 5)

  ; h|d
  [
    [0.5 0.4 0.1]
    [0.5 0.6 0.9]
   ]

(require '[criterium.core :as c])

(print-cause-trace *e)

(let
    [model
       {:fg
        (fgtree
          (:v [:pv [0.5 1]]
            [:s|v [[0.5 0.5] [[0.5 0.7] [0.7 2]]]
              (:s [:ps [1/2 4]])
             ]))
        :priors {:v :pv :s :ps}}
     ]
  (->>
    (reductions
      (fn update-it [{{s :s} :marginals :as m} {v :pv :as data}]
        (let [p [(or (:mu (meta s)) 0.5) (or (:sigma (meta s)) 4)]]
          (update-priors (assoc m :data (assoc data :ps p)))))
        model
       (concat (repeat 8 {:pv [0 0.1]}) (repeat 8 {:pv [1 0.1]})))
      (map (comp :s :marginals))
      rest
    ;(map (fn [f] (map (juxt identity f) (range 0 1.25 0.25))))
      ((fn [sfs]
         (let [n (count sfs) n1 (/ 1 n)]
           (view
            (xy-chart
              (map-indexed
                (fn [i f]
                  [(str i)
                   {:x (range 0 1 0.01)
                    :y (map f (range 0 1 0.01))
                    :style {:line-color (Color. ^float (* n1 i) 0.0 ^float (- 1.0 (* n1 i)) 0.5)
                            :marker-type :none
                            :line-style :solid}}]) sfs)
              {:title "-"
               :x-axis {:title "specificity"}
               :y-axis {:title "p" :decimal-pattern "##.##"}
               :theme :matlab})))))
      ))

  (require '[com.hypirion.clj-xchart :as xc :refer [view xy-chart]])

  (import '[java.awt Color])


  (em/make-zero 0)

  (defprotocol Qq :extend-via-metadata true (q [this x]))

  (defprotocol Rq (r [this x]))

  (defrecord R [t]
    Rq (r [t x] (* 5 x)))

  (deftype RR [] Rq (r [t x] (* 5 x)))

  (let [t {:x 5 :y 7 :f (fn [x] (* 5 x))}
        im (with-meta t {`q (fn [this x] ((:f this) x))})]
    (c/quick-bench (q im 6)))

  (def t1 {:x 5 :y 7 :f (fn [x] (* 5 x))})
  (def im (with-meta t1 {`q (fn [this x] ((:f this) x))}))

  (c/quick-bench (q im 6))

  (let [t (R. 8)]
    (c/quick-bench (r t 6)))

  (let [t (RR.)]
    (c/quick-bench (r t 6)))

  (-> (R. 4 5 6) (assoc :w 4))

  (:x :y)

  Rq


  (ns-unmap 'sigmapi.core 'make-node)
  (ns-unmap 'sigmapi.normal 'make-node)

  (ns-unmap 'sigmapi.normal 'make-node)

  "

  V>< :h ([1.3219280948873622 1.3219280948873622 2.321928094887362])
  V>< :d ([##Inf -0.0])
  F>< :h|d ([##Inf -0.0])
  p: [[##Inf 1.0] [##Inf 0.7369655941662062] [##Inf 0.15200309344504997]]  s: [1.0 0.7369655941662062 0.15200309344504997]
  V>< :h ([1.0 0.7369655941662062 0.15200309344504997])
  V>< :h ([1.3219280948873622 1.3219280948873622 2.321928094887362])
  F>< :h|d ([1.3219280948873622 1.3219280948873622 2.321928094887362])
  p: [[2.321928094887362 2.6438561897747244 5.643856189774724] [2.321928094887362 2.0588936890535683 2.473931188332412]]  s: [1.3959286763311392 0.6896598793878492]
  V>< :d ([1.3959286763311392 0.6896598793878492])
  V>< :h ([1.3219280948873622 1.3219280948873622 2.321928094887362] [1.0 0.7369655941662062 0.15200309344504997])
  V>< :d ([##Inf -0.0] [1.3959286763311392 0.6896598793878492])
  V>< :h ([1.6322682154995132 1.369233809665719 1.784271308944563])
  V>< :d ([##Inf -0.0])
  F>< :h|d ([##Inf -0.0])
  p: [[##Inf 1.0] [##Inf 0.7369655941662062] [##Inf 0.15200309344504997]]  s: [1.0 0.7369655941662062 0.15200309344504997]
  V>< :h ([1.0 0.7369655941662062 0.15200309344504997])
  V>< :h ([1.6322682154995132 1.369233809665719 1.784271308944563])
  F>< :h|d ([1.6322682154995132 1.369233809665719 1.784271308944563])
  p: [[2.632268215499513 2.6911619045530815 5.106199403831925] [2.632268215499513 2.1061994038319254 1.936274402389613]]  s: [1.534657418873091 0.6107884880890615]
  V>< :d ([1.534657418873091 0.6107884880890615])
  V>< :h ([1.6322682154995132 1.369233809665719 1.784271308944563] [1.0 0.7369655941662062 0.15200309344504997])
  V>< :d ([##Inf -0.0] [1.534657418873091 0.6107884880890615])



  "



)