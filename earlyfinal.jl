using NNlib: NNlib
using Umlaut
using Graphs, AttributeGraphs
using GraphMakie, GLMakie

# FIXME Piracy
copyifpresent(@nospecialize(x)) = ismissing(x) ? x : copy(x)
Base.copy(ag::AttributeGraph) = AttributeGraph(
    copyifpresent(getgraph(ag)),
    copyifpresent(vertex_attr(ag)),
    copyifpresent(edge_attr(ag)),
    copyifpresent(graph_attr(ag))
)

# TODO there should be a better way to do this
const OP_LIST = Base.IdSet{Function}([
    Base.materialize,
    Base.:*,
    Base.:+,
    NNlib.conv,
    NNlib.maxpool,
    NNlib.meanpool
])

"""
    resolve_func(op::Umlaut.Call)

Finds the actual function being called for a call op.
Returns a tuple `(func, is_kwcall)` to allow for detection of kwcalls.
"""
function resolve_func(op::Umlaut.Call)
    fn = op.fn
    while fn isa Umlaut.V
        res = op.tape[fn]
        if res isa Umlaut.Constant
            fn = res.val
        elseif res isa Umlaut.Call
            fn = res.fn
        elseif res isa Umlaut.V
            fn = res
        else
            error("Unexpected op found during variable lookup: $res")
        end
    end
    fn === Core.kwcall && return (op.args[2], true)
    return (fn, false)
end


"""
    isaliasing(op::Umlaut.Call)

Returns whether a given call's return value aliases one or more of its inputs (e.g. by creating a wrapper type).
Currently assumes no arguments escape from a function call outside of being incorporated into the return value.
"""
function isaliasing(op::Umlaut.Call)
    fn, _ = resolve_func(op)
    return !(fn in OP_LIST)
end

function find_uses(tape::Umlaut.Tape)
    uses = OAttributeGraph(DiGraph(); vertex_type=Umlaut.V, edge_type=Bool)
    foreach(_ -> addvertex!(uses), 1:length(tape))
    for (var, op) in enumerate(tape.ops)
        addvertexattr!(uses, var, Umlaut.V(op))
        op isa Umlaut.Call || continue
        for arg in op.args
            if arg isa Umlaut.V
                addedge!(uses, arg.id, var)
                addedgeattr!(uses, arg.id, var, isaliasing(op))
            end
        end
    end
    uses
end

"""
    last_uses!(uses::AttributeGraph)

Collapses use chains such that variables are connected to their last (possibly transitive) use.
For example, a sequence of `A -aliasing-> B -non-aliasing-> C` will be collapsed to `A -non-aliasing-> C`.
Finding the last use allows us to ensure finalizer calls are only inserted after all possible aliases of
the original value (and their aliases, transitively) are no longer live.
"""
function last_uses!(uses::AttributeGraph)
    function isterminal(from, to)
        getedgeattr(uses, from, to) || return true  # Non-aliasing uses are always terminal
        # So are dest nodes without outgoing edges (i.e. nobody depends on them)
        return isempty(outneighbors(uses, to)) # && uses[to] != tape.result
    end

    for var in vertices(uses)
        # Iteration stops at a fixed point where all uses (outgoing edges) either:
        # 1. are non-aliasing
        # 2. are aliasing but not themselves used anywhere
        while any(v -> !isterminal(var, v), outneighbors(uses, var))
            for use in outneighbors(uses, var)
                isnonterm = getedgeattr(uses, var, use)
                if isnonterm && !isempty(outneighbors(uses, use))
                    for use2 in outneighbors(uses, use)
                        addedge!(uses, var, use2)
                        addedgeattr!(uses, var, use2, getedgeattr(uses, use, use2))
                    end
                    # Edge from source var -> direct use needs to be removed last because
                    # we need the edge attribute (alias/non-alias info) above
                    remedge!(uses, var, use)
                end
            end
        end
    end
    return uses
end


"""
    add_finalizers!(tape::Umlaut.Tape, uses::AttributeGraph)

Inserts calls to `finalize` for each array value of interest after it and its aliases are no longer live.
Currently, "array values of interest" are limited to return values of known non-aliasing ops.
"""
function add_finalizers!(tape::Umlaut.Tape, uses::AttributeGraph)
    # Two-pass algorithm because the uses graph stores absolute tape positions,
    # and those would be invalidated if we add new ops to the tape as we go.
    array_vars = Dict{Umlaut.V,Umlaut.V}()
    for op in tape.ops
        if op isa Umlaut.Call && resolve_func(op)[1] in OP_LIST
            array_var = Umlaut.V(op)
            last_use = Umlaut.V(tape, maximum(outneighbors(uses, array_var.id)))
            # TODO insert finalizers for args if return value is non-aliasing call?
            if last_use != tape.result
                array_vars[array_var] = last_use
            end
        end
    end

    # @show array_vars
    for (array_var, last_use) in pairs(array_vars)
        insert!(tape, last_use.id + 1, mkcall(Base.finalize, array_var))
    end
    return tape
end

show_uses(uses::AttributeGraph) = graphplot(
    uses;
    edge_color=[getedgeattr(uses, e.src, e.dst) ? :red : :black for e in edges(uses)],
    ilabels=repr.(1:nv(uses)),
    arrow_shift=:end
)

# ---- Test ----

using Flux
using Yota: GradCtx

# Copied from MLUtils
function flatten(x::AbstractArray{<:Any,N}) where {N}
    return reshape(x, :, size(x, N))
end

# Copied from Metalhead
function alexnet(; dropout_prob=0.5, inchannels::Integer=3, nclasses::Integer=1000)
    backbone = Chain(
        Conv((11, 11), inchannels => 64, relu; stride=4, pad=2),
        MaxPool((3, 3); stride=2),
        Conv((5, 5), 64 => 192, relu; pad=2),
        MaxPool((3, 3); stride=2),
        Conv((3, 3), 192 => 384, relu; pad=1),
        Conv((3, 3), 384 => 256, relu; pad=1),
        Conv((3, 3), 256 => 256, relu; pad=1),
        MaxPool((3, 3); stride=2)
    )
    classifier = Chain(
        AdaptiveMeanPool((6, 6)), flatten,
        Dropout(dropout_prob),
        Dense(256 * 6 * 6, 4096, relu),
        Dropout(dropout_prob),
        Dense(4096, 4096, relu),
        Dense(4096, nclasses)
    )
    return Chain(backbone, classifier)
end

loss(m, x, y) = Flux.mse(m(x), y)

tape, uses = let
    model = alexnet(; nclasses=10)
    x = rand(Float32, 224, 224, 3, 64)
    y = rand(Float32, 10, 64)

    @time begin
        res, tape = Umlaut.trace(loss, model, x, y; ctx=GradCtx())
    end
    uses = find_uses(tape)
    # show_uses(uses)

    open(io -> show(io, tape), "experiments/tape_before.txt", "w")
    open(io -> show(io, Umlaut.to_expr(tape)), "experiments/expr_before.jl", "w")

    last_uses!(uses)
    # show_uses(uses)

    add_finalizers!(tape, uses)
    open(io -> show(io, tape), "experiments/tape_after.txt", "w")
    open(io -> show(io, Umlaut.to_expr(tape)), "experiments/expr_after.jl", "w")

    tape, uses
end;
