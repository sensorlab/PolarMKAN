import gdown
import os

ManyTx = "https://drive.google.com/uc?id=17EnvGFoflJEh1xhFC8wx5fhCuPYhWt2l"

ManySig_LINK = "https://drive.google.com/uc?id=1szuns8MhcYocdbipK9t9TM9MLgEMklxk"

# Download WiSig dataset
gdown.download(ManySig_LINK, 'many_sig', quiet=False)
os.system('unzip many_sig')

gdown.download(ManyTx, 'many_tx', quiet=False)
os.system('unzip many_tx')

# Download Oracle dataset
#ORACLE_LINK = 'https://repository.library.northeastern.edu/downloads/neu:m044q5210?datastream_id=content'
#os.system(f'wget {ORACLE_LINK}')