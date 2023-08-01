# ClustroDrug
Drug repurposing, or drug repositioning, has gained increasing attention as a strategy to identify new therapeutic applications for existing drugs, which can expedite the drug discovery process by leveraging existing knowledge of drug safety and pharmacokinetics.

This is **ClustroDrug**, a novel unsupervised clustering algorithm for drug repurposing that leverages High Throughput Screening data, specifically the Area Under the Curve (AUC) values obtained from large-scale screening of drugs. 

ClustroDrug utilizes a two-tier approach with statistical and machine learning techniques to cluster drugs based on their AUC profiles, which reflect their pharmacological activities across multiple assays or biological targets to identify clusters of drugs with shared characteristics that may be indicative of common therapeutic uses.

The first tier of the algorithm identifies clusters of drugs with similarity, defined by metrics such as **Pearson correlation coefficient**, above a certain threshold. After obtaining a group of rough clusters, the second tier applies **Markov Clustering Algorithm** to refine and enhance the clustering structure of the graph by identifying densely connected regions. 

The hyperparameters of the model, such as the similarity cutoff, are fitted by the model that optimizes **Normalized Mutual Information** score across a public screening data of 1500 FDA approved drugs over 500 cell lines. Then, those hyperparameters are used to cluster drugs in the lab data to provide information and interrelationship about these lesser-known drugs. 

Our results demonstrate that ClustroDrug exhibits robust performance in accurately clustering drugs with similar molecular properties, suggesting its potential for identifying repurposing opportunities.

For further information, you can access my research paper here:
https://www.researchgate.net/publication/372824376_ClustroDrug_a_Novel_Algorithm_for_Clustering_Drugs_based_on_High_Throughput_Screening_Data_to_Repurpose_Drugs

Visuals representing Drug Clusters:

<img width="1430" alt="Screen Shot 2023-08-01 at 16 21 28" src="https://github.com/BerkeAltiparmak/ClustroDrug/assets/96665962/7b30a515-ef4e-4e93-9495-c8f0680d39de">

<img width="690" alt="Screen Shot 2023-08-01 at 16 22 05" src="https://github.com/BerkeAltiparmak/ClustroDrug/assets/96665962/bac6d812-c588-4171-86ca-31564f47823d">

<img width="748" alt="Screen Shot 2023-08-01 at 16 22 42" src="https://github.com/BerkeAltiparmak/ClustroDrug/assets/96665962/d7ebcb30-dd27-4ac3-8568-e8ca5a1f5812">


