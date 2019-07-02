
import os
database='databasehere'
model='modelhere'
directorio='./pretrain_models/'+database+'/'+model+'/'

for folder in os.listdir(directorio):
        for f in os.listdir(directorio+folder):
                if os.path.isfile(directorio+folder+"/"+f+"/.nandetected"):
                        print "Nan file detected at {}".format(directorio+folder+"/"+f)
                elif os.path.isfile(directorio+folder+"/"+f+"/.expnotfinished"):
                        print "Model did not finished at {}".format(directorio+folder+"/"+f)

