import socket
import os

def data_root_path():
    ''' Defining the different root path using the host name '''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
#     if 'nid' in hostname:
#         rootpath = '/scratch/snx3000/nperraud/' 
#     elif 'omenx' in hostname:
#         rootpath = '/store/nati/datasets/audio/'         
#     else:
    if 1:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = os.path.join(utils_module_path, '../../data/')
    return rootpath