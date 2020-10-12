module SeqData

include("seq.jl")
include("masked_seq.jl")
include("joinedseqs.jl")

export SequentialData, AbstractSeqData, MaskedSequentialData, JoinedSeqData

end # module
