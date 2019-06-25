# EEG Classification using Deep and Transfer Learning

#### Requirements 

Dependencies can be found in:
* requirements.txt 
* environment.yml (for conda virt env)

#### Data:

Download EEG MI datasets from:
* BCI Competition IV 2a: http://www.bbci.de/competition/iv/#dataset2a
* High-Gamma: https://web.gin.g-node.org/robintibor/high-gamma-dataset

#### Create dirs:

In the root of the project:
* Make a directory for the (processed) data: "/data"
* Make a directory for the results: "/results"

#### Data preprocessing:
To preprocess the data got to "/data_loader/process_data/.." 
For both datasets there is separate preprocess script.

#### Config the experiment:
In "config.yaml" (in the project root) you can configure the experiment.
* experiment->dataset: [hgd, bciciv2a]
* model->name: [eegnet, deep, shallow]
* experiment->type: [no_tl, loo_tl, ccsa_da]
    *no_tl: no transfer learning, just dataset+CNN model
    *loo_tl: leave-one-out transfer leaning
    *ccsa_da: feature space alignment, siamese network with contrastive loss,
    ccsa: classification and contrastive semantic alignment loss
* server->full_cv: True=trains model for all subjects 4-fold (sequentially), 
False=trains only model for subject_id and i_valid_fold given.

#### Run experiment:
Run the file "/experiment/run.py" to start the experiment.
Results get put into "/results/" dir.



:bowtie: :tophat: