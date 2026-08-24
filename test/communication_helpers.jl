@inline test_communicator() = LatticeMatrices._default_communicator()
@inline test_comm_size() = LatticeMatrices._comm_size(test_communicator())
@inline test_comm_rank() = LatticeMatrices._comm_rank(test_communicator())
@inline test_allreduce_sum(value) =
    LatticeMatrices._allreduce_sum(value, test_communicator())
