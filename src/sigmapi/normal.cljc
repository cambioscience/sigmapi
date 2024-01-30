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
  ([{:keys [graph id mu sigma dim-for-node] :as params}]
   (let [f ((if (number? mu) normal multivariate-normal) mu sigma)
         s1 (:sigma-1 (meta f))
         d  (if (number? s1) 0 (em/num-cols s1))
         zv (em/make-zero 1 d)
         zm (em/make-zero d)
         node
         (with-meta {:f f :id id :dim-for-node dim-for-node :kind :factor :features #{:passes}
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
              (fn [this {:keys [mu sigma] :as p}]
                (assoc this :f ((if (number? mu) normal multivariate-normal) mu sigma)))
            })] node)))

(defmethod make-node [:map :factor :normal]
  ([{:keys [graph id clm cpm dim-for-node] :as params}]
   (let [f (apply (if (number? (first cpm)) normal multivariate-normal) cpm)
         s1 (:sigma-1 (meta f))
         d (if (number? s1) 0 (em/num-cols s1))
         zv (em/make-zero 1 d)
         zm (em/make-zero d)
         node
         (with-meta {:f f :id id :dim-for-node dim-for-node :kind :factor :features #{:passes}
                     :zero-vector zv :zero-matrix zm :d d}
           {; sp/Messaging
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



)