import logging
import random
import mhcflurry
from mhcflurry.hyperparameters import HyperparameterDefaults
from mhcflurry.class1_allele_specific.cross_validation import cross_validation_folds
from mhcflurry.class1_allele_specific.train import (train_across_models_and_folds, 
    HYPERPARAMETER_DEFAULTS)

def from_list_to_dict(peptides, affinities):
    """
    Returen a peptide->affinity dictionary from dataframe or peptide list
    """
    peptide_to_affinity_dict = {}
    for idx, peptide in enumerate(peptides):
        peptide_to_affinity_dict[peptide] = affinities[idx]
    
    return peptide_to_affinity_dict

def get_mhc_pseudo(file):
    """
    Return a allele->sequence dictionary
    """
    mhc_pseudo = {}
    with open(file,'r') as f:
        for line in f.readlines():
            if len(line.strip().split()[0]) == 10:
                mhc_pseudo[line.strip().split()[0]] = line.strip().split()[1]

    return mhc_pseudo 
    
def get_mhc_pep_seq(mhc_pseudo, peptides, hla):
    """
    Return input list with MHC pseudo-sequence attached
    """
    peptides_long = []
    for peptide in peptides:
        assert len(mhc_pseudo[hla]) == 34, "mhc sequence length not correct"
        peptides_long.append(peptide + mhc_pseudo[hla])

    return peptides_long

def get_mhc_pep_seq_rd(mhc_pseudo, peptides):
    """
    Return input list with RANDOM MHC pseudo-sequence attached
    """
    peptides_long = []
    for peptide in peptides:
        #assert len(mhc_pseudo[hla]) == 34, "mhc sequence length not correct"
        peptides_long.append(peptide + mhc_pseudo[random.choice(mhc_pseudo.keys())])

    return peptides_long

def hyperparameter_selection(epoch):
    """
    Return models with different sets of hyperparameters
    """
    models = HYPERPARAMETER_DEFAULTS.models_grid(
        activation=["relu"],layer_sizes=[[16],[32],[64]],optimizer=["adam"],
        loss=["binary_crossentropy"],embedding_output_dim=[16,32],
        dropout_probability=[.1,.2,.25],n_training_epochs=[epoch],impute=[False]
        )
    
    return models

def hyperparameter_selection_score_by_cv(models = None, fold=3, 
        data=None, alleles=[None]):
    """
    Return the hyperparameter set with the best combination score
    """
    cv_folds = cross_validation_folds(data,
        n_folds=fold,drop_similar_peptides=False,
        alleles=alleles)
    logging.warning(
        "Training %d model architectures across %d folds = %d models"
        % (len(models),len(cv_folds),len(models) * len(cv_folds)))
    cv_results = train_across_models_and_folds(
        cv_folds,models)
    cv_results["summary_score"] = (
        cv_results.test_auc.fillna(0) +
        cv_results.test_tau.fillna(0) +
        cv_results.test_f1.fillna(0))
    model_ranks = (
        cv_results.ix[cv_results.allele == alleles[0]]
            .groupby("model_num")
            .summary_score
            .mean()
            .rank(method='first', ascending=False, na_option="top")
            .astype(int)).to_dict()
    cv_results["summary_rank"] = [
        model_ranks[row.model_num] for (_, row) in cv_results.iterrows()
    ]
    cv_results.to_csv("cv_results.txt",header=True,index=False)
    best_architectures_by_allele = (
        cv_results.ix[cv_results.summary_rank == 1]
            .set_index("allele")
            .model_num
            .to_dict())
    
    return best_architectures_by_allele
