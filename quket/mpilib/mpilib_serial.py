SUM = 'SUM'
MAX = 'MAX'
def Get_processor_name():
    return 'dummy'
def Finalize():
    pass
class COMM_WORLD:
    def Get_size():
        return 1
    def Get_rank():
        return 0
    def Bcast(buf, root):
        return None
    def bcast(buf, root):
        return buf
    def Allreduce(sendbuf, recvbuf, op):
        raise Exception("This function is not meant for no MPI version!")
    def allreduce(sendbuf, op):
        return sendbuf
    def Gather(sendobj, recvobj, root):
        raise Exception("This function is not meant for no MPI version!")
    def gather(sendobj, root):
        return sendobj
    def send(obj, dest, tag): 
        pass
    def recv(buf, source, tag): 
        return None
    def barrier():
        pass
    def Barrier():
        pass
    def Split(color, key):
        #return self
        pass
    def Free():
        pass

