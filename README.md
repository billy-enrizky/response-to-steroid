# Multimodal AI prediction

```mermaid
flowchart TD
    %% Pathology-slide pipeline
    PS([Pathology Slides]) --> UE[UNI Encoder]
    UE --> SEL1{Train & Test<br/>5 Models}
    SEL1 --> RF1[Random Forest]
    SEL1 --> SVM1[SVM]
    SEL1 --> GBT1[Gradient Boosting Tree]
    SEL1 --> LP1[Linear Probing]
    SEL1 --> KNN1[KNN Probing]
    RF1 --> ProbImg[Probabilities<br/>best model: SVM]
    SVM1 --> ProbImg
    GBT1 --> ProbImg
    LP1 --> ProbImg
    KNN1 --> ProbImg
    
    %% Clinical-data pipeline
    CD([Clinical Data]) --> PREP[Data Preprocessing]
    PREP --> IMP[Data Imputation]
    IMP --> SEL2{Train & Test<br/>4 Models}
    SEL2 --> RF2[Random Forest]
    SEL2 --> SVM2[SVM]
    SEL2 --> GBT2[Gradient Boosting Tree]
    SEL2 --> LOGR[Logistic Regression]
    RF2 --> ProbClin[Probabilities<br/>best model: Random Forest]
    SVM2 --> ProbClin
    GBT2 --> ProbClin
    LOGR --> ProbClin
    
    %% Late fusion
    ProbImg -->|w = 0.5| FUS[Late Fusion]
    ProbClin -->|w = 0.5| FUS
    FUS --> PRED[Final Prediction]


```mermaid
flowchart TD
    subgraph Data_Preparation
        A1[Load Pathology Images] --> A2[Train-Test Split]
        A2 --> A3[Create Image Datasets]
    end

    subgraph Feature_Extraction
        B1[UNI Encoder] --> B2[Extract Image Features]
        B2 --> B3[Save Features to File]
    end

    subgraph Model_Evaluation
        C1[Load Extracted Features] --> C2[Evaluate Multiple Models]
        C2 --> |Linear Probe|C3[Model Performance]
        C2 --> |KNN|C3
        C2 --> |ProtoNet|C3
        C2 --> |SVM|C3
        C2 --> |Random Forest|C3
        C2 --> |Gradient Boosting|C3
        C2 --> |Logistic Regression|C3
        C3 --> C4[Select Best Model]
    end

    subgraph Image_Analysis
        D1[Top-k Patch Retrieval] --> D2[Visualize Top Patches]
        D3[Slide-level Aggregation] --> D4[Slide-level Evaluation]
    end

    subgraph Clinical_Analysis
        E1[Load Clinical Data] --> E2[Preprocess & Impute]
        E2 --> E3[Engineer Features]
        E3 --> E4[Train Clinical Models]
        E4 --> |Random Forest|E5[Clinical Model Performance]
        E4 --> |SVM|E5
        E4 --> |Gradient Boosting|E5
        E4 --> |Logistic Regression|E5
    end

    subgraph Multimodal_Fusion
        F1[Match Patients] --> F2[Extract Image & Clinical Probabilities]
        F2 --> F3[Dimension-Weighted Late Fusion]
        F3 --> F4[Compare Modalities]
    end

    %% Main flow connections
    A_Preparation([Data Preparation]) --> Data_Preparation
    B_Extraction([Feature Extraction]) --> Feature_Extraction
    C_Evaluation([Model Evaluation]) --> Model_Evaluation
    D_Analysis([Image Analysis]) --> Image_Analysis
    E_Clinical([Clinical Analysis]) --> Clinical_Analysis
    F_Fusion([Multimodal Fusion]) --> Multimodal_Fusion

    Data_Preparation --> Feature_Extraction
    Feature_Extraction --> Model_Evaluation
    Model_Evaluation -->|Best Model| Image_Analysis
    Clinical_Analysis --> Multimodal_Fusion
    Image_Analysis --> Multimodal_Fusion
    Multimodal_Fusion --> G[Final Performance Comparison]
