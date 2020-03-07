using SeqData




A = rand(10, 400)
#(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Real, N_valid::Real; verbose=false, supervised=false)

# test unsupervised
train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true)

@assert train[1][:,:,1] == A[:,1:64]
train[end]
@assert train[2][:,:,1] == A[:,1+24:24+64]


# test supervised

train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true, supervised=true)

@assert train[1][1][:,:,1] == A[:,1:64]
@assert train[1][2][:,:,1] == A[:,2:65]
@assert train[2][1][:,:,1] == A[:,1+24:24+64]


B = rand(10,10, 400)

train, valid, test = SequentialData(B, 24, 64, 0.5, 0.25, verbose=true)

@assert train[1][:,:,:,1] == B[:,:,1:64]
train[end]
@assert train[2][:,:,:,1] == B[:,:,1+24:24+64]


# test supervised

train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true, supervised=true)

@assert train[1][1][:,:,1] == A[:,1:64]
@assert train[1][2][:,:,1] == A[:,2:65]
@assert train[2][1][:,:,1] == A[:,1+24:24+64]





train, valid, test = SequentialData(A, 0, 64, 0.5, 0.25, verbose=true)

@assert train[1] == A[:,1:64]
train[end]
@assert train[2] == A[:,2:65]


true
