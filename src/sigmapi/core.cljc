(ns ^{:doc "𝝨𝝥" :author "Matthew Chadwick"}
  sigmapi.core
  "

    Implementations of the sum-product and max-sum
    algorithms, from Factor Graphs and the Sum-Product Algorithm
    Frank R. Kschischang, Senior Member, IEEE, Brendan J. Frey, Member, IEEE, and
    Hans-Andrea Loeliger, Member, IEEE
    IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 47, NO. 2, FEBRUARY 2001
    DOI: 10.1109/18.910572

    Also, Pattern Recognition and Machine Learning,
    Christopher M. Bishop, was invaluable




    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    /*
    * Copyright (C) 2016 Intermine
    *
    * This code may be freely distributed and modified under the
    * terms of the GNU Lesser General Public Licence. This should
    * be distributed with the code. See the LICENSE file for more
    * information or http://www.gnu.org/copyleft/lesser.html.
    *
    */


    TODO:

    * rewrite this with transducers
    * separate api from impl
    * Neanderthal implementation

  "
  (:require
    [clojure.core.matrix :as m]
    [clojure.set :as set]
    [loom.graph :as lg]
    [loom.alg :as la]
    [clojure.walk :as walk])
    #?(:cljs (:require-macros
              [sigmapi.core :refer [fgtree]])
       :clj (:import [mikera.vectorz Vector])))

#?(:clj
  (defmacro fgtree [xp]
   (walk/postwalk
     (fn [x]
       (if (and (seqable? x) (keyword? (first x)))
         `(~(if (vector? x) `vector `list) ~@x)
         x))
     xp)))

(defn log [x]
  #?(:clj  (Math/log x)
     :cljs (js/Math.log x)))

(defn pow [x y]
  #?(:clj  (Math/pow x y)
     :cljs (js/Math.pow x y)))

(def log2 (log 2))

(defn ln [x] (/ (log x) log2))

(def ln- (comp (partial * -1) ln))

(defn P [x] (pow 2 (* -1 x)))

(defn normalize
  ([p]
    (normalize p (reduce + p)))
  ([p s]
   (if (zero? s)
    p
    (vec (map (partial * (/ 1 s)) p)))))

(defn random-matrix
  "Returns a random matrix of the given shape e.g.  [2 3 4 5]"
  [[f & r]]
  (if (nil? r)
    (repeatedly f rand)
    (repeatedly f (partial random-matrix r))))

(comment
  "
    Protocols
    There are 2 types of node in a factor graph: Variable and Factor
    There are several algorithms that can be run on a factor graph,
    each of which causes different kinds of messages to be exchanged.
    The only constant is the messaging itself, and that each node must
    have product (x) and identity (i) functions. Product doesn't mean
    multiplication necessarily, it just means the ability to combine
    messages into one.
  ")

(defprotocol Messaging
  "

    Messaging

    >< return an inflowing (leaves-first) message for the given
       node-id to from the the given messages
    <> return an outflowing (root-first) message for the given
       node-id to from the the given messages

    i return the identity message for this node


    messages always excludes the destination node


  "
  (>< [this messages to])
  (<> [this messages to to-msg parent])
  (i [this]))

; Variable nodes
(defprotocol Variable)

; Factor nodes
(defprotocol Factor)

; (Factor) nodes that pass when given the opportunity to be root
(defprotocol Passes
  (pass? [this]))

(defprotocol LogSpace
  "Return a probability for x"
  (p [this x]))

(defn indexed-best
  "
    Returns list of the best (according to the given function f)
    items in the given matrix, and their indices.
  "
  [f]
  (fn ibf [mat]
    (let [best (f mat)]
      [best (first (filter #(= best (apply m/mget mat %))
            (m/index-seq mat)))])))

(def indexed-min (indexed-best m/emin))

(def indexed-max (indexed-best m/emax))

(defn rotate-vec
  "
    Rotate the given vector v so that index
    i is at the position given by fn f (first or last)
  "
  [v i f]
  (if (= f last)
    (into (subvec v (inc i) (count v)) (subvec v 0 (inc i)))
    (into (subvec v i (count v)) (subvec v 0 i))))

(defn tranz
  "
    Return the given matrix mat transposed such that the dimension at index i
    is the first or last (f) dimension.
    e.g. if m is a matrix with shape [2 3 4 5], then (tranz m 2 last) has shape [5 2 3 4]
          or (tranz m 2 first) has shape [4 5 2 3]
    Don't like this, there must be a better way.
  "
  ([mat i f]
    (tranz mat (vec (range (m/dimensionality mat))) i f))
  ([mat v i f]
   (let [rv (rotate-vec v i f)
         tm (m/transpose mat rv)
         ]
    [tm rv
      (rotate-vec v
        (mod (- (dec (m/dimensionality mat)) i) (m/dimensionality mat))
        (if (= last f) first last))])))

; mat is a matrix, g is the operation for combining matrices
(defn combine
  "
  Returns the product of the given messages
  using the given function g (e.g. add).
  Each message's value will have a different dimension
  so the matrix is transposed so that its last dimension
  matches. (Broadcast the message's vector would still involve
  transposing)
  Finally the result is transposed so the dimension of
  the destination node to is the first dimension, ready
  for summing. Hmm maybe that last bit should be a separate fn
  "
  [mat g messages to dim-for-node]
  (let
    [
     dimz (vec (range (m/dimensionality mat)))
     [p pv ddd]
       (reduce
         (fn [[r pv pnd] {id :id v :value}]
           (let [
                 d (get pnd (dim-for-node id))
                 [tm rv nd] (tranz r dimz d last)
                 tm #?(:clj
                       (cond-> tm (not (instance? Vector tm)) m/matrix)
                       :cljs tm)
                 q (g tm v)]
             [q rv (mapv pnd nd)]))
         [mat dimz dimz] messages)
       d (get ddd (dim-for-node to))
       [tm rv nd] (tranz p dimz d first)
     ]
    tm))

(deftype FactorNode
  [f id dim-for-node]
  Messaging
  (>< [this messages to]
    (let [
          prod (combine f m/add messages to dim-for-node)
          sum (m/emap ln- (map m/esum (m/emap P prod)))
          ]
      {
       :value     sum
       :repr      (cons '∑ (list (cons '∏ (list (:repr (i this)) (if (== 1 (count messages)) (:repr (first messages)) (map :repr messages))))))
       }))
  (<> [this messages to to-msg parent-msg]
    (>< this messages to))
  (i [this]
    {:value f :repr id :dim-for-node dim-for-node})
  Factor
  Passes
  (pass? [this] true)
  LogSpace
  (p [this x] (m/emap P x)))


(comment "
	Returns a factor node for the max-sum algorithm,
	for the given function f (a matrix), id and
	map of node-id-to-dimensions.
	This node operates in negative log space.
	")
(deftype MaxFactorNode
  [f id dim-for-node]
  Messaging
  (>< [this messages to]
    (let [
          rsum (combine f m/add messages to dim-for-node)
          mm (map m/emin rsum)
          ]
      {
       :dim-for-node dim-for-node
       :value        mm
       :min          (indexed-min mm)
       :sum          rsum
       :im           (vec (map (fn [[s c]] [s (zipmap (keys (dissoc dim-for-node to)) c)]) (map indexed-min rsum)))
       :repr         (list 'min (cons '∑ (cons (:repr (i this)) (map :repr messages))))
       }))
  (<> [this messages to to-msg parent-msg]
    (let [
          conf (get-in parent-msg [:configuration id])
          mind (zipmap (map :id messages) (range (count messages)))
          to-conf (get conf to)
          ]
      {
       :dim-for-node dim-for-node
       :value        0
       :mind					mind
       :conf conf
       :configuration (assoc (:configuration parent-msg) to to-conf)
       }))
  (i [this] {:value f :repr id :dim-for-node dim-for-node})
  Passes
  (pass? [this] true)
  Factor
  LogSpace
  (p [this x] (m/emap P x)))

(comment "Returns a variable node for the
max-sum algorithm with the given id")
(deftype MaxVariableNode
  [id]
  Messaging
  (>< [this messages to]
    (let [sum (apply m/add (map :value messages))]
      {
       :value     sum
       :repr      (cons '∑ (map :repr messages))
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
       :value         sum
       :min           min
       :configuration (assoc configuration to mto)
       :repr          (cons '∑ (map :repr messages))
       }))
  (i [this] {:value 0 :repr 0})
  Variable
  Passes
  (pass? [this] false)
  LogSpace
  (p [this x] (m/emap P x)))

(deftype VariableNode [id]
  Messaging
  (>< [this messages to]
    {
     :value     (apply m/add (map :value messages))
     :repr      (if (== 1 (count messages)) (:repr (first messages)) (cons '∏ (map :repr messages)))
     })
  (<> [this messages to to-msg parent-msg]
    (>< this messages to))
  (i [this]
    {:value 0 :repr id})
  Variable
  Passes
  (pass? [this] false)
  LogSpace
  (p [this x] (m/emap P x)))

(defmulti make-node
  (fn [{:keys [alg type] :as p}]
    [alg type]))

(defmethod make-node [:sp/sp :sp/factor]
  ([{:keys [graph id clm cpm dfn]}]
    (FactorNode. (or clm (m/emap ln- cpm)) id dfn)))

(defmethod make-node [:sp/mxp :sp/factor]
  ([{:keys [graph id clm cpm dfn]}]
    (MaxFactorNode. (or clm (m/emap ln- cpm)) id dfn)))

(defmethod make-node [:sp/sp :sp/variable]
  ([{id :id}]
    (VariableNode. id)))

(defmethod make-node [:sp/mxp :sp/variable]
  ([{id :id}]
    (MaxVariableNode. id)))

(defn neighbourz [edges]
  (reduce
    (fn [r [{a :id} {b :id}]]
      (-> r
        (update a (fn [n] (conj (or n []) b)))
        (update b (fn [n] (conj (or n []) a)))))
    {} edges))

(defn leaf?
  "Is the given node of the given graph a leaf node
  this depends on whether the graph is directed"
  ([g n]
   (== 1 (lg/out-degree g n))))

(defn leaves [g]
  (filter
    (partial leaf? g)
    (lg/nodes g)))

(defn edges->fg
  "
  TODO: need to check shape of graph and
  dimensionality of matrices
  "
  ([alg edges]
      (let [g (apply lg/graph (map (partial map :id) edges))
            t (lg/graph (la/bf-span g (:id (ffirst edges))))
            nodes (into {} (map (juxt :id identity) (mapcat identity edges)))
            neighbours (neighbourz edges)
            ]
        {
         :alg        alg
         :messages   {}
         :graph      g
         :spanning-tree t
         :leaves     (leaves t)
         :neighbours neighbours
         :nodes
                     (into {}
                       (map
                         (fn [id]
                           [id (if-let [mat (get-in nodes [id :matrix])]
                                 (make-node {:alg alg :type :sp/factor :graph g :id id
                                             :cpm (m/matrix mat)
                                             :dfn (zipmap (neighbours id) (range))
                                             :mfn (zipmap (neighbours id) (map #(get-in nodes [% :matrix]) (neighbours id)))})
                                 (make-node {:alg alg :type :sp/variable :id id}))])
                         (lg/nodes g)))})))

(defn matrices-as-vectors [fg]
  (reduce
    (fn [r [id mat]]
      (assoc r id
        {
          :shape  (m/shape mat)
          :vector (m/as-vector mat)
        }))
    {} (filter (partial satisfies? Factor) (:nodes fg))))

(defn update-factors
  "Replace nodes for the given matrices with new ones"
  ([model matrices]
    (update-factors model matrices :cpm))
  ([{g :graph alg :alg nodes :nodes :as model} matrices cmkey]
    (reduce
     (fn [model [id mat]]
       (let [n (nodes id) {dfn :dim-for-node} (i n)]
         (assoc-in model [:nodes id]
           (make-node {:alg alg :type :sp/factor :graph g :id id cmkey (m/matrix mat) :dfn dfn}))))
     model matrices)))

(defn change-alg
  "
  Replace nodes for the given matrices with new ones
  TODO: I'm probably going to replace the OO-style Variable & Factor nodes
  with a multimethod that dispatches on the alg and node type
  "
  [{g :graph alg :alg nodes :nodes :as model}]
  (reduce
    (fn [model [id node]]
      (let [{dfn :dim-for-node v :value} (i node)]
        (assoc-in model [:nodes id]
         (make-node {:alg   alg :type (if (satisfies? Variable node) :sp/variable :sp/factor)
                     :graph g :id id :clm v :dfn dfn}))))
    (assoc model :messages {}) nodes))

; make this work with one edge
(defn as-edges
  ([exp]
    (as-edges exp []))
  ([exp edges]
    (if (and (seqable? exp) (keyword? (first exp)) (or (keyword? (first (first (rest exp)))) (keyword? (first (second (rest exp))))))
      (let [branches (if (vector? exp) (rest (rest exp)) (rest exp))]
        (reduce
          (fn [r c]
              (as-edges c
                (conj r
                  [(let [f {:id (first exp)}] (if (vector? exp) (assoc f :matrix (second exp)) f))
                   (if (vector? c)
                    {:id (first c) :matrix (second c)}
                    {:id (first c)})])))
          edges branches))
      edges)))

(defn exp->fg
  "Return a factor graph for the given expression"
  [alg exp]
  (edges->fg alg (as-edges exp)))

(defn prior-nodes [{:keys [graph nodes] :as model}]
  (into {} (map (fn [id] [id (nodes id)]) (filter
                                    (fn [n]
                                      (and (leaf? graph n) (satisfies? Factor (nodes n))))
                                    (lg/nodes graph)))))

(defn msgs-from-leaves [{:keys [messages graph nodes] :as model}]
  (reduce
    (fn [r id]
      (let [parent (first (lg/successors graph id))]
        (assoc-in r [:messages parent id]
          (assoc (i (get nodes id))
            :id id :flow :><))))
    model (leaves graph)))

(defn msgs-from-variables [{:keys [messages graph nodes] :as model}]
  (reduce
    (fn [r id]
      (let [parent (first (lg/successors graph id))]
        (assoc-in r [:messages parent id]
          (assoc (i (get nodes id))
            :id id :flow :><))))
    model
    (filter (comp (fn [n] (satisfies? Variable n)) nodes)
      (lg/nodes graph))))

(comment
  "

    Frey2001Factor DOI: 10.1109/18.910572 page 502

    As in the single-i algorithm, message passing is initiated at
    the leaves. Each vertex v remains idle until messages have arrived
    on all but one of the edges incident on v. Just as in the
    single-i algorithm, once these messages have arrived, v is able
    to compute a message to be sent on the one remaining edge
    to its neighbor (temporarily regarded as the parent), just as in
    the single-i algorithm, i.e. according to Fig. 5. Let us denote
    this temporary parent as vertex w. After sending a message to w
    , vertex v returns to the idle state, waiting for a “return message”
    to arrive from w. Once this message has arrived, the vertex
    is able to compute and send messages to each of its neighbors
    (other than w), each being regarded, in turn, as a parent.


    Bishop, Pattern Recognition and Machine Learning page 412

    We can readily generalize this result to arbitrary tree-structured
    factor graphs by substituting the expression (8.59) for the factor
    graph expansion into (8.89) and again exchanging maximizations with products.
    The structure of this calculation is identical to that of the
    sum-product algorithm, and so we can simply translate those
    results into the present context.
    In particular, suppose that we designate a particular
    variable node as the ‘root’ of the graph.
    Then we start a set of messages propagating inwards from the leaves
    of the tree towards the root, with each node sending its message
    towards the root once it has received all incoming messages from its other neighbours.
    The final maximization is performed over the product
    of all messages arriving at the root node,
    and gives the maximum value for p(x).
    This could be called the max-product algorithm and is
    identical to the sum-product algorithm except that summations
    are replaced by maximizations. Note that at this stage,
    messages have been sent from leaves to the root,
    but not in the other direction.

    (somewhere in one paper it says that any node can be root,
    actually one node will emerge as root somewhere naturally but
    if it's a factor node it needs to pass and allow a variable
    node to be the root)

  "
  )

(defn message-passing
  "

    Synchronous message-passing on the given model given previous-model.
    This is the simplest form of message-passing. Can add other
    kinds (loopy, asynchronous core.async) later but this is easy to test

    Returns the given model with its messages updated

    A model is a map with:

    * a graph describing its structure
    * a messages map of the form {:node-to-id {:node-from-id message}}
    * a map of nodes (variable, factor)

    Messaging has two phases: in from the leaves >< and out
    from the root <>. The root is the variable node that messages
    from the leaves happen to converge on (if a factor node
    happens to get the chance to become root it passes).
    Each node combines its messages according to its local
    product and sum functions and sends the result to the node
    it's summarizing for
    (or all but one of its neighbours if it's root - see comment above).

  "
  [previous-model {:keys [messages graph nodes] :as model}]
  (reduce
    (fn [{root? :root :as r} [id msgs]]
      (let [prev-msgs (get-in previous-model [:messages id]) node (get nodes id)]
        ; messages have arrived on all but one of the edges incident on v
        (if (and (not= msgs prev-msgs) (== (count msgs) (dec (lg/out-degree graph id))))
         (let [parent (first (set/difference (lg/successors graph id) (into #{} (keys msgs))))
               node (get nodes id)]
           (assoc-in r [:messages parent id]
             (assoc (>< node (vals (dissoc msgs parent)) parent)
              :flow :>< :id id)))
         ; all messages have arrived
         (if (and (not= msgs prev-msgs) (== (count msgs) (lg/out-degree graph id)))
           (let [[return _] (first (set/difference
                                        (into #{} (map (juxt :id :flow) (vals msgs)))
                                        (into #{} (map (juxt :id :flow) (vals prev-msgs)))))]
             (if (and (pass? node) (= :>< (get-in msgs [return :flow])))
               (if root? r (update-in r [:messages id] dissoc return))
               (reduce
                 (fn [r parent]
                   (assoc
                     (assoc-in r [:messages parent id]
                       (assoc (<> node (vals (dissoc msgs parent)) parent (get msgs parent)
                          (if root? (get msgs return) nil))
                          :flow :<> :id id))
                     :root id))
                 r (keys (if root? (dissoc msgs return) msgs)))))
           r))))
    model messages))

(defn can-message?
  "The algorithm terminates once two messages have been passed
  over every edge, one in each direction."
  [{:keys [messages graph nodes] :as model}]
  (or (empty? messages)
      (not= (into #{} (keys messages))
            (into #{} (mapcat keys (vals messages))))))

(defn propagate
  "Propagate messages on the given model's graph
  in both directions"
  ([m]
    (propagate message-passing (assoc m :messages {})))
  ([f m]
    (last
     (last
       (take-while (comp can-message? first)
         (iterate (fn [[o n]] [n (f o n)])
           [m (msgs-from-leaves m)]))))))

(defn propagate-cycles
  "Propagate messages on the given model's graph
  in both directions"
  ([m n]
    (propagate-cycles message-passing n (assoc m :messages {})))
  ([f n m]
    (last
     (last
       (take n
         (iterate (fn [[o n]] [n (f o n)])
           [m (msgs-from-leaves m)]))))))

(defn maybe-list [x]
  (if (seqable? x) x (list x)))

(defn normalize-vals [m]
  (into {}
    (map
      (juxt key
        (comp (fn [v] (if (== 1 (m/dimensionality v)) (normalize v) (vec (map normalize v)))) val)) m)))

(defn marginals
  "Returns a map of marginals for the nodes of the given model"
  [{:keys [messages graph nodes] :as model}]
  (into {}
    (map
      (fn [[id node]]
        [id (vec (m/emap P (maybe-list (:value (<> node (vals (get messages id)) nil nil nil)))))])
      (filter (comp (fn [n] (satisfies? Variable n)) val) nodes))))

(defn all-marginals
  "Marginals for all given models"
  [models]
  (reduce
    (fn [r m] (merge-with conj r m))
      (zipmap
        (map key
             (filter
               (comp (fn [n] (satisfies? Variable n)) val)
               (:nodes (first models))))
        (repeat [])) (map marginals models)))

(defn configuration
  "Returns the total configuration of max-sum for the given model"
  [{:keys [messages graph nodes] :as model}]
  (reduce
    (fn [r c] (merge r c))
    {} (map (comp :configuration val)
            (filter (comp (partial = :<>) :flow val)
                    (mapcat val
                      (select-keys messages (leaves graph)))))))

(defn filter-configuration [config {:keys [messages graph nodes] :as model}]
  (select-keys config
    (filter (fn [id] (= MaxVariableNode (type (get nodes id))))
      (keys config))))

(def MAP-config
  "Returns the MAP configuration of the given model"
  (comp (partial apply filter-configuration)
     (juxt configuration identity)))

(defn compute-marginals [exp]
  (normalize-vals
    (marginals (propagate (exp->fg :sp/sp exp)))))

(defn compute-MAP-config [exp]
  (MAP-config
    (propagate (exp->fg :sp/mxp exp))))

(defn as-states [config model]
  (into {}
    (map
      (fn [[k v]] [k (get-in model [:states k v])])
      config)))

(defn check-leaves [config sequence-by-id]
  (into {}
    (map
      (fn [[id sequence]] [id (= sequence (get config id))])
      sequence-by-id)))

(defn print-msgs [{:keys [messages graph nodes] :as model}]
  (doseq [[to from msg] (mapcat (fn [[to msgs]] (map (partial cons to) msgs)) messages)]
    (println from "⟶" to (:flow msg) (:repr msg)
             "sum: " (m/shape (:sum msg))
             "dfn: " (:dim-for-node msg)
             "conf" (:configuration msg)
             "mind" (:mind msg)
             "min: " (:min msg))
    (println "    " "val: " (:value msg) (m/shape (:im msg)) "im: " (:im msg))
    (println "    " "min" (:min msg) "conf" (:configuration msg))))

(defn msgs [{:keys [messages graph nodes] :as model}]
  (map
    (fn ([[to from msg]] {:from from :to to :repr (:repr msg)}))
    (mapcat (fn [[to msgs]] (map (partial cons to) msgs)) messages)))

(defn msg-diff [om nm]
  (reduce
    (fn [r [to msgs]]
      (assoc-in r [:messages to]
                (let [diff (set/difference (into #{} (vals msgs)) (into #{} (vals (get-in om [:messages to]))))]
                  (zipmap (map :id diff) diff))))
    nm (:messages nm)))

(defn learn-variables [graph post priors data]
  (reductions
    (fn [[g post] data-priors]
      (let [
              p2 (select-keys post (keys priors))
              p1 (merge (zipmap (vals priors) (map p2 (keys priors))) data-priors)
              g  (update-factors g p1)
            ]
        [g (normalize-vals (marginals (propagate g)))]))
    [graph (or post (zipmap (keys priors) (map (comp (partial mapv P) :value i (:nodes graph)) (vals priors))))] data))

(defn learned-variables [{:keys [fg learned marginals priors data] :as model}]
  (let [[g m]
          (last
           (learn-variables
             (or learned (exp->fg :sp/sp fg)) marginals priors data))]
    (-> model
      (assoc :marginals m)
      (assoc :learned g))))

(defn learn-step
  [{:keys [fg learned marginals priors data] :as model}]
      (let [
             {nodes :nodes :as graph} (or learned (exp->fg :sp/sp fg))
              post (or marginals (zipmap (keys priors) (map (comp (partial map P) :value i nodes) (map (fn [v] (if (keyword? v) v (last v))) (vals priors)))))
              p2 (select-keys post (keys priors))
              p1 (merge (zipmap (map (fn [v] (if (keyword? v) v (first v))) (vals priors)) (map p2 (keys priors))) data)
              g (update-factors graph p1)
            ]
        (-> model
          (assoc :learned g)
          (assoc :marginals (normalize-vals (sigmapi.core/marginals (propagate g)))))))
