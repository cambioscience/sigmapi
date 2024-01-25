(ns sigmapi.normal
  (:require
    [clojure.stacktrace :refer [print-cause-trace]]
    [clojure.math :as maths :refer [PI]]
    [clojure.core.matrix :as m]
    [kixi.stats.distribution :as xd]
    [emmy.env :as e :refer :all]
    [emmy.matrix :as em]
    [sigmapi.core :as sp :refer :all]))

(defn mup [mat rs cs f]
  (m/set-selection mat rs cs
    (f (m/select mat rs cs))))

(defn qr
  "
    QR decomposition by Householder reflection

    adapted from Matlab implementation by @tobydriscoll
    returns Q the eigenvectors and R
    the diagonal of which is the eigenvalues
    Qd = I 0
         0 F
    F begins at column d, row d
    F is a n-d dimensional vector space
    in which a hyperplane H reflects z
    to the vector |z|e1
    v = |z|e1 - z
    Fy = (I - 2(vv'/v'v))y
    which is an orthonormal projector
  "
  ([a]
   (qr (m/identity-matrix (first (m/shape a))) (m/shape a) a))
  ([I [m n] A]
    (loop [d 0 R A Q I]
      (if (< d n)
        (let [[z1 :as z] (m/select R (range d m) d)
              v (m/matrix
                  (cons
                    (- (* -1.0 (Math/signum (double z1)) (m/magnitude z)) z1)
                    (m/mul -1.0 (m/select z :rest))))
              Qd (mup I (range d m) (range d n)
                    (fn Fy [i] (m/sub i (m/mul 2.0 (m/div (m/outer-product v v)
                                                          (m/inner-product v v))))))]
          (recur (inc d) (m/mmul Qd R) (m/mmul Q Qd)))
        {:A A
         :Q Q
         :R R
         :A=QR (m/mmul Q R)
         :eigenvectors (m/mul -1 (m/transpose Q))
         :eigenvalues (m/mul -1 (m/diagonal R))
         }))))

(defn rotation-matrix
  "make a (column-based) rotation matrix from these angles"
  [[x y z]]
  (->
    [
       (* (cos x) (cos y))
       (* (sin x) (cos y))
       (* -1.0 (sin y)) 0

       (- (* (* (cos x) (sin y)) (sin z)) (* (sin x) (cos z)))
       (+ (* (* (sin x) (sin y)) (sin z)) (* (cos x) (cos z)))
       (* (cos y) (sin z)) 0

       (+ (* (* (cos x) (sin y)) (cos z)) (* (sin x) (sin z)))
       (- (* (* (sin x) (sin y)) (cos z)) (* (cos x) (sin z)))
       (* (cos y) (cos z)) 0

       0 0 0 1
    ]
    (m/reshape [4 4])))

(defn scale-matrix [[x y z]]
  (m/matrix
    [
     [x 0 0 0]
     [0 y 0 0]
     [0 0 z 0]
     [0 0 0 1]
     ]))

(defn translation-matrix [[x y z]]
  (m/matrix
    [
     [1 0 0 x]
     [0 1 0 y]
     [0 0 1 z]
     [0 0 0 1]
     ]))

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
               x (apply em/row x)
               x-mu (em/transpose (- x mu))
               det-sigma (em/determinant sigma)]
           (/
             (exp (get-in (* -1/2 (em/transpose x-mu) sigma-1 x-mu) [0 0]))
             (sqrt (* (expt (* 2 PI) k) det-sigma)))))
       {:mu mu :sigma sigma :sigma-1 sigma-1})))

(defn conditional-mvn-a|b&c [mu sigma]
  (fn [a]
    (let [[u1 u2] mu
         [S11 S12 S21 S22] sigma
          mu_ (+ u1 (* S12 (em/invert S22) (- a u2)))
          sigma_ (- S11 (* S12 (em/invert S22) S21))
         ]
      (multivariate-normal mu_ sigma_))))

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
  ([{:keys [graph id mu sigma dfn] :as params}]
   (let [f ((if (number? mu) normal multivariate-normal) mu sigma)
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
              (fn [this {:keys [mu sigma]}]
                (assoc this :f ((if (number? mu) normal multivariate-normal) mu sigma)))
            })] node)))

(defmethod make-node [:map :factor :normal]
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
                 :dim-for-node dim-for-node
                 :sum mu
                 :value 1
                 :min (apply max-key first (map vector mu (range)))
                 :im (mapv (fn [[s c]] [s (zipmap (keys (dissoc dim-for-node to)) c)]) mu)
                 :repr (list 'min (cons '∑ (cons (:repr (i this)) (map :repr messages))))
                 }))
            `<> (fn [{:keys [f id dim-for-node zero-vector zero-matrix d] :as this} messages to to-msg parent-msg]
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
            `i (fn [{:keys [f id dim-for-node zero-vector zero-matrix d]}]
                 {:value f :repr id :dim-for-node dim-for-node})
            ; Updatable
            `updated
            (fn [this {:keys [cpm]}]
              (assoc this :f (apply (if (number? (first cpm)) normal multivariate-normal) cpm)))
            })] node)))

(defmethod make-node [:map :variable :normal]
  ([{:keys [id] :as node}]
   (with-meta node
     {; Messaging
     `><
      (fn [{:keys [f id dim-for-node d] :as this} messages to]
        {
         :value (product-of-normals (assoc this :to-dim (dim-for-node to) :messages messages))
         :repr (cons '∑ (map :repr messages))
         })
      `<> (fn [{:keys [f id dim-for-node d] :as this} messages to to-msg parent-msg]
            (let
               [
                ; to-msg is the msg received by this node from to on the >< pass,
                ; which contains the indices of the other variables for each of this variable's states.
                ; Here we are telling to its configuration and the configurations of all previous variables
                ; In the outflowing messaging, the root variable node uses all its messages
                {:keys [mu sigma] :as prod} (product-of-normals (assoc this :to-dim (dim-for-node to) :messages (cons to-msg messages)))
                ;sum (apply m/add (map :value (cons to-msg messages)))
                maxv (apply max-key first (map vector mu (range)))
                ; look up the configuration we got in the forward pass which lead to this minimum
                ; (for the root - others need to use the indices they got from the parent)
                conf (if parent-msg (get-in parent-msg [:configuration id]) (get-in maxv [1]))
                configuration (if parent-msg (:configuration parent-msg) {id conf})
                mto (get-in to-msg [:im conf 1])
                ]
              {
               :value sum
               :min maxv
               :configuration (assoc configuration to mto)
               :repr (cons '∑ (map :repr messages))
               }))
      `i
      (fn [this]
        {:value (with-meta (fn identity [x] (em/by-rows [1])) {:mu 0 :sigma 1 :sigma-1 1}) :repr id})})))




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


  (vector? (first {:x 5}))

(require '[criterium.core :as c])

(print-cause-trace *e)

(let
    [model
       {:fg
        (fgtree
          (:s0 [:ps0 {:mu 0.5 :sigma 2}]
            [:s1|s0&v
             {:mu [-0.3 0.2 0.2]
              :sigma
              (m/to-nested-vectors
                  (m/submatrix
                   (m/mmul
                     (scale-matrix [1.9 2.3 4.3])
                     ;(scale-matrix [0.5 0.5 0.5])
                     (rotation-matrix [-0.4 -0.65 0.2])
                     ) 0 3 0 3))
              }
             (:v [:pv {:mu 0.5 :sigma 1}])
             (:s1)]))
        :priors {:v :pv :s0 :ps0}
        :impl :normal}
     ]
  (->>
    (reductions
      (fn update-it [{{s :s1} :marginals :as m} {pv :pv :as data}]
        (let [p {:mu (or (:mu (meta s)) 0.5) :sigma (or (:sigma (meta s)) 4)}]
          (println " >" p)
          (update-priors (assoc m :data (assoc data :ps0 p)))))
      model
      (concat
        (repeat 16 {:pv {:mu 1 :sigma 0.1}})
        (repeat 16 {:pv {:mu 0 :sigma 0.1}})
        ;(repeat 4 {:pv [0 0.1]})
         ;(repeat 4 {:pv [1 0.1]})
         ))
      (map (comp :s1 :marginals))
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

  (qr
    [[1.00 0.60 0.99]
     [0.60 1.00 0.10]
     [0.99 0.10 1.00]])

  (qr
    [[5 0]
     [0 1]])

  (let [t (m/to-nested-vectors
            (m/submatrix
             (m/mmul
               (rotation-matrix [0.32 -0.66 -0.17])
               (scale-matrix [1.936 2.389 4.386])
               (scale-matrix [1 1 -1])
               ) 0 3 0 3))]
    t)



  (/ 2 (sqrt 13))

  (/ -3.60555127546399 -1.386750490563073)

  ((multivariate-normal
     [0.5 0.5]
     [[2.0 0.5]
      [0.5 2.0]]) [1 0])

  (require '[com.hypirion.clj-xchart :as xc :refer [view xy-chart]])

  (import '[java.awt Color])

  (require '[cljplot.render :as pr]
           '[cljplot.build :as pb]
           '[clojure2d.color :as pc]
    '[cljplot.core :refer [save show]]
    '[fastmath.random :as fr])

  (fr/randval 0 1)

  (fr/grand)

  (fr/randval [(fr/grand) (fr/grand)] [(fr/grand -10 1) (fr/grand -10 1)])

  (let [mu [0.5 0.5 0.5]
        a 0.9 b 0.9 c 0.1
        sigma [[1.0 a b]
               [a 1.0 c]
               [b c 1.0]]
        f (multivariate-normal mu sigma)
        fn2d (fn [v s'] (f [v s' 0.5]))
        ;ms (summarize mu sigma 2)
        ]
    (-> (pb/series [:function-2d fn2d {:x [0 1] :y [0 1]}])
     (pb/preprocess-series)
     (pb/add-axes :bottom)
     (pb/add-axes :left)
     (pb/add-label :bottom "2d function")
     (pr/render-lattice {:width 512 :height 512})
     (save "results/examples/function2d.jpg")
     (show)))

  (let [mu [0 0]
        sigma [[3 2]
               [2 3]]
        f (multivariate-normal mu sigma)
        fn2d (fn [v s'] (f [v s']))
        ;ms (summarize mu sigma 2)
        ]
    (-> (pb/series [:function-2d fn2d {:x [-10 10] :y [-10 10]}])
     (pb/preprocess-series)
     (pb/add-axes :bottom)
     (pb/add-axes :left)
     (pb/add-label :bottom "2d function")
     (pr/render-lattice {:width 512 :height 512})
     (save "results/examples/function2d.jpg")
     (show)))

  (let [mu [0.5 0.5]
        sigma [[1.0 0.9]
               [0.9 1.0]]
        f (multivariate-normal mu sigma)
        {:keys [mu sigma]} (meta f)
        fn2d (fn [v s'] (f [v s']))
        ms (summarize mu sigma 0)
        ]
    (view
      (xy-chart
        [["p"
          {:x (range -10 10 0.01)
           :y (map ms (range -10 10 0.01))
           :style {:line-color :black
                   :marker-type :none
                   :line-style :solid}}]]
        {:title "-"
         :x-axis {:title "specificity"}
         :y-axis {:title "p" :decimal-pattern "##.##"}
         :theme :matlab})))


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

  (apply max-key first (map vector [1 52 21] (range)))

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