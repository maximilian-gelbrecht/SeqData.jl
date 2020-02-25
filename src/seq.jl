using Flux

abstract type AbstractSeqData end

"""
    SequentialData

Struct with initialization for preparing sequence data for a learning problem. It saves a lot of memory because it uses Julia's iterator interface and does not save the same data multiple times as a naive implementation would do. The default initialization splits the data into a train, valid and test set.

`SequentialData` can be used for `for`-loops with `for i in data` and also indexed with data[i]. Each element is a 2-element array containing the input and output for the learning problem.

If more than one trajectory is provided, each trajectory will be divided in train, valid and test sets and these sets will then be combined to form the full train, valid and test sets.

It is possible to output batches of data.

# Fields of the struct

* `data::AbstractArray`: Raw Data as a ``N_dim \\times N_t \\times N_i`` array
* `N_batch::Int`: Batch Size
* `N_length::Int`: Length of each sample
* `N::Int`: Total number of individual train/valid/test data pairs available
* `N_i::Int`: Number of Input Trajectories
* `N_tr_i::Int`: Number of individual train/valid/test data pairs available per trajectory
* `N_t::Int`: Number of time steps per trajectory

# Initialization

    SequentialData(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Int, N_valid::Int; supervised::Bool=false)

## Input

* `input_data::AbstractArray`: Raw data in either a ``N_dim \\times N_t \\times N_i`` or ``N_dim \\times N_t`` array
* `N_batch::Int`: Batch size, If `N_batch==0` the output will be 2D without any batching
* `N_length::Int`: Length of each sample
* `N_train::Real`: Relative Number of input samples for training
* `N_valid::Real`: Relative Number of input samples for validation. All remaining time steps will form the test set
* `supervised::Bool`: If true every indexing will return a pair of arrays where the second array is shifted by 1 from the first array.


If ``N_i > 1`` all values are refering to each trajectory seperately.

## Output

Returns _three_ instances of `SequentialData` in order `(train_set, valid_set, test_set)`.

"""
struct SequentialData <: AbstractSeqData
    data::AbstractArray
    N_batch::Int
    N_length::Int
    N::Int
    N_i::Int
    N_tr_i::Int
    N_t::Int
    _batches::Bool
    _supervised::Bool
end

# initializes all data
function SequentialData(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Real, N_valid::Real; verbose=false, supervised=false)

    @assert 0. <= N_train <= 1.
    @assert 0. <= N_valid <= 1.
    @assert N_train + N_valid <= 1.
    @assert 0 <= N_batch

    _batches = N_batch == 0 ? false : true
    N_batch = N_batch == 0 ? 1 : N_batch
    #offset = supervised ? 0 : 1

    # convert data to   N_i x N_t x N_dim
    if ndims(input_data)==2
        N_dim, N_t = size(input_data)
        input_data = reshape(input_data, (N_dim, N_t, 1))
        N_i = 1
    elseif ndims(input_data)==3
        N_dim, N_t, N_i = size(input_data)
    else
        error("input_data has to be 2- or 3-dimensional in (N_dim x N_t x N_i) format")
    end

    N_train = Int(ceil(N_t * N_train))
    N_valid = Int(ceil(N_t * N_valid))

    N_Ntrain = Int(ceil((N_train - N_length)/N_batch)) #- N_batch ??
    N_Nvalid = Int(ceil((N_valid - N_length)/N_batch))
    N_Ntest = Int(ceil(((N_t - N_train - N_valid) - N_length)/N_batch))

    if (N_valid - N_length) < 0
        @warn "Valid set may be empty"
    end

    if (N_t - N_train - N_valid - N_length) < 0
        @warn "Test set may be empty"
    end

    if verbose
        println("Train set length = ",N_i*N_Ntrain)
        println("Valid set length = ",N_i*N_Nvalid)
        println("Test set length = ",N_i*N_Ntest)
    end

    SequentialData(input_data[:,1:N_train,:], N_batch, N_length, N_i*N_Ntrain, N_i, N_Ntrain, N_train, _batches, supervised), SequentialData(input_data[:,N_train+1:N_train+N_valid,:], N_batch, N_length, N_i*N_Nvalid, N_i, N_Nvalid, N_valid, _batches, supervised), SequentialData(input_data[:,N_train+N_valid+1:end,:], N_batch, N_length, N_i*N_Ntest, N_i, N_Ntest, N_t - N_train - N_valid, _batches, supervised)
end

function Base.iterate(iter::AbstractSeqData, state=1)
    if state>iter.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::AbstractSeqData) = iter.N
Base.eltype(iter::AbstractSeqData) = Array{typeof(iter.data),1}

function Base.getindex(iter::SequentialData, i::Int)
    @assert 1 <= i <= iter.N

    i_tr = Int(ceil(i/iter.N_tr_i)) # which trajectory
    ii = i - (i_tr - 1)*iter.N_tr_i # where on this trajectory are we?
    N_batch = _batch_size(iter, ii)

    if !iter._supervised
        data = gpu(zeros(eltype(iter.data), size(iter.data,1), iter.N_length, N_batch))

        for (iib, ib) in enumerate(_batch_iterate_range(iter, ii))
            data[:,:,iib] = iter.data[:,ib:ib-1+iter.N_length,i_tr]
        end

        if iter._batches
            return data
        else
            return data[:,:,1]
        end
    else
        data1 = gpu(zeros(eltype(iter.data), size(iter.data,1), iter.N_length, N_batch))
        data2 = gpu(zeros(eltype(iter.data), size(iter.data,1), iter.N_length, N_batch))

        for (iib, ib) in enumerate(_batch_iterate_range(iter, ii))
            data1[:,:,iib] = iter.data[:,ib:ib-1+iter.N_length,i_tr]
            data2[:,:,iib] = iter.data[:,ib+1:ib+iter.N_length,i_tr]
        end

        if iter._batches
            return (data1, data2)
        else
            return (data1[:,:,1], data2[:,:,1])
        end

    end
end


"""
    _batch_size(data::AbstractSeqData, i::Int)

Outputs the batch size of the current batch. This is either the regular batch size that was chosen or for the last batch of each trajectory all possibly remaining pairs.

Input `i` is the index on the _current trajectory_.
"""
function _batch_size(data::AbstractSeqData, i::Int)
    @assert 1 <= i <= data.N_tr_i
    if i != data.N_tr_i
        N_batch = data.N_batch
    else
        N_batch = data.N_t - (i-1)*data.N_batch - data.N_length
        @assert 1 <= N_batch <= data.N_batch
    end
    return N_batch
end

"""
    _batch_iterate_range(data::AbstractSeqData, i::Int)

Outputs the range over all parts of a batch. This is either the regular batch size that was chosen or for the last batch of each trajectory all possibly remaining pairs.

Input `i` is the index on the _current trajectory_.
"""
function _batch_iterate_range(data::AbstractSeqData, i::Int)
    if i != data.N_tr_i # all but the last batch per trajectory
        iterate_range = (data.N_batch*(i-1))+1:i*data.N_batch
    else  # last batch per trajectory might not have N_batch individuals
        iterate_range = (data.N_batch*(i-1))+1:(data.N_t - data.N_length)
    end
end

Base.firstindex(iter::AbstractSeqData) = 1
Base.lastindex(iter::AbstractSeqData) = iter.N

"""
    time_index(data::AbstractSeqData, i::Int)

Given an instance of the train, valid or predict set, returns the time step of the forecasted value, so the last value of the output of the model and the index of the trajetory the time step is from (in case only one trajectory is used as input data this is always 1.)

"""
function time_index(data::AbstractSeqData, i::Int)
    @assert 1 <= i <= data.N

    i_tr = Int(ceil(i/data.N_tr_i)) # which trajectory

    ii = i - (i_tr - 1)*data.N_tr_i # where on this trajectory are we?

    N_batch = _batch_size(data, ii)
    indices = zeros(Int, (N_batch, 2))
    indices[:,1] .= i_tr
    indices[:,2] = collect(_batch_iterate_range(data, ii)) .+ (data.N_length+data.offset-1-data.hybrid_offset)

    return indices
end
