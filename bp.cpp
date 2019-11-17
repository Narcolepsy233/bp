//2019 NUAA 161730115 --Author Shaoyu Yang (Narcolepsy233@gmail.com)
#include <iostream>
using namespace std;
#include <cmath>
#include <ctime>
#include <fstream>

#define INPUT_LAYER_NEURE_NUM 3 //由于没有人为的添加截距项1，所以输入向量值最后应当为一，实际输入神经元为该值减一
#define HIDDEN_LAYER_NEURE_NUM 2
#define OUTPUT_LAYER_NEURE_NUM 3
#define ETA 0.5 //学习率
#define K 1     //输入样本向量个数
#define INPUTDATAFILENAME "test.txt"
#define DODATAFILENAME "do.txt"

//简单bp神经网络，只含有一层隐含层，S曲线使用Sigmoid函数

void ReadFileInput(double inputVector[INPUT_LAYER_NEURE_NUM][K]){
    fstream ofile;
    ofile.open(INPUTDATAFILENAME,ios::in);
    if(ofile.fail()){
        cout<<"Failed to open file"<<endl;
        exit(0);
    }
    for(int i=0;i<INPUT_LAYER_NEURE_NUM;i++){
        for(int j=0;j<K;j++){
            if(ofile.eof()){
                cout<<"Input Num Not enough"<<endl;
                exit(0);
            }
            ofile>>inputVector[i][j];
        }
    }
    cout<<"--  InputVector Data input over  --"<<endl;
}

void ReadFileDo(double Do[OUTPUT_LAYER_NEURE_NUM][K]){
    fstream ofile;
    ofile.open(DODATAFILENAME,ios::in);
    if(ofile.fail()){
        cout<<"Failed to open file"<<endl;
        exit(0);
    }
    for(int i=0;i<INPUT_LAYER_NEURE_NUM;i++){
        for(int j=0;j<K;j++){
            if(ofile.eof()){
                cout<<"Input Num Not enough"<<endl;
                exit(0);
            }
            ofile>>Do[i][j];
        }
    }
    cout<<"--  Antipate Data input over  --"<<endl;
}


float getRand(){
    srand(int(rand()));
    return 2.0*rand()/RAND_MAX -1.0;
}
//---------------------------------------------------------Initialize
void InitializeWho(double who[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM]){
    for(int i=0;i<OUTPUT_LAYER_NEURE_NUM;i++){
        for(int j=0;j<HIDDEN_LAYER_NEURE_NUM;j++){
            who[i][j]=getRand();
            // cout<<who[i][j]<<" ";
        }
        // cout<<endl;
    }
}
void InitializeWhi(double wih[HIDDEN_LAYER_NEURE_NUM][INPUT_LAYER_NEURE_NUM]){
    for(int i=0;i<HIDDEN_LAYER_NEURE_NUM;i++){
        for(int j=0;j<INPUT_LAYER_NEURE_NUM;j++){
            wih[i][j]=getRand();
            // cout<<wih[i][j]<<" ";
        }
        // cout<<endl;
    }
}
void Initialize_bh(double bh[HIDDEN_LAYER_NEURE_NUM]){
    for(int i=0;i<HIDDEN_LAYER_NEURE_NUM;i++){
        bh[i]=getRand();
    }
}
void Initialize_bo(double bo[OUTPUT_LAYER_NEURE_NUM]){
    for(int i=0;i<OUTPUT_LAYER_NEURE_NUM;i++){
        bo[i]=getRand();
    }
}
void Initialization(double wih[HIDDEN_LAYER_NEURE_NUM][INPUT_LAYER_NEURE_NUM],
    double who[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM],
    double bh[HIDDEN_LAYER_NEURE_NUM],
    double bo[OUTPUT_LAYER_NEURE_NUM],double &eps,int &M){
    InitializeWhi(wih);
    InitializeWho(who);
    Initialize_bh(bh);
    Initialize_bo(bo);

    cout<<"Please input eps: ";
    cin>>eps;
    cout<<"Please input M: ";
    cin>>M;
    cout<<"-- Initialization finished --"<<endl;
}
//--------------------------------------------------
double getSquare(double x){
    return x*x;
}

double Efunc(double Do[OUTPUT_LAYER_NEURE_NUM][K],double yo[OUTPUT_LAYER_NEURE_NUM],int k){
    double result=0;
    for(int i=0;i<OUTPUT_LAYER_NEURE_NUM;i++){
        result+=getSquare(Do[i][k]-yo[i]);
    }
    return result/2;
}

double ffunc_Sigmoid(double x){
    return 1/(1+exp(-x));
}

void UpdateWho(double yo[OUTPUT_LAYER_NEURE_NUM],
    double who[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM],
    double Do[OUTPUT_LAYER_NEURE_NUM][K],
    double ho[HIDDEN_LAYER_NEURE_NUM],
    double partial_ihEO[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM],int k)
    {
    double partial_EO=0;
    double partial_OI=0;
    double partial_Iw=0;
    double partial_Ew=0;
    for(int i=0;i<OUTPUT_LAYER_NEURE_NUM;i++){
        partial_EO=-(Do[i][k]-yo[i]);
        partial_OI=yo[i]*(1-yo[i]);
        for(int j=0;j<HIDDEN_LAYER_NEURE_NUM;j++){
            partial_Iw=ho[j];
            partial_Ew=partial_EO*partial_Iw*partial_OI;
            partial_ihEO[i][j]=partial_EO*partial_OI*who[i][j];
            who[i][j]-=ETA*partial_Ew;
        }
    } 
}

void UpdateWih(double ho[HIDDEN_LAYER_NEURE_NUM],
    double wih[HIDDEN_LAYER_NEURE_NUM][INPUT_LAYER_NEURE_NUM],
    double hi[HIDDEN_LAYER_NEURE_NUM],
    double InputVector[INPUT_LAYER_NEURE_NUM][K],
    double partial_ihEO[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM], int k
    ){
    double partial_allEO=0;
    double partial_OI=0;
    double partial_Iw=0;
    double partial_Ew=0;
    for(int i=0;i<HIDDEN_LAYER_NEURE_NUM;i++){
        for(int l=0;l<OUTPUT_LAYER_NEURE_NUM;l++){
            partial_allEO+=partial_ihEO[l][i];
        }
        partial_OI=ho[i]*(1-ho[i]);
        for(int j=0;j<INPUT_LAYER_NEURE_NUM;j++){
            partial_Iw=InputVector[j][k];
            partial_Ew=partial_allEO*partial_OI*partial_Iw;
            wih[i][j]-=ETA*partial_Ew;
            // cout<<wih[i][j]<<" ";
        }
        // cout<<endl;
    }

}

double training(//一轮学习过程
    double inputVector[INPUT_LAYER_NEURE_NUM][K],
    double Do[OUTPUT_LAYER_NEURE_NUM][K],
    double wih[HIDDEN_LAYER_NEURE_NUM][INPUT_LAYER_NEURE_NUM],
    double who[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM],
    double bh[HIDDEN_LAYER_NEURE_NUM],
    double bo[OUTPUT_LAYER_NEURE_NUM],int k)
    {//选取第K组数据
    
    double E=0;//全局误差
    double hi[HIDDEN_LAYER_NEURE_NUM];
    double ho[HIDDEN_LAYER_NEURE_NUM];
    double yi[OUTPUT_LAYER_NEURE_NUM];
    double yo[OUTPUT_LAYER_NEURE_NUM];
    double partial_ihEO[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM];

    for(int i=0;i<HIDDEN_LAYER_NEURE_NUM;i++){
        for(int j=0;j<INPUT_LAYER_NEURE_NUM;j++){
            hi[i]+=inputVector[j][k]*wih[i][j];
        }
        hi[i]-=bh[i];
    }

    for(int i=0;i<HIDDEN_LAYER_NEURE_NUM;i++){
        ho[i]=ffunc_Sigmoid(hi[i]);
    }

    for(int i=0;i<OUTPUT_LAYER_NEURE_NUM;i++){
        for(int j=0;j<HIDDEN_LAYER_NEURE_NUM;j++){
            yi[i]+=ho[j]*who[i][j];
        }
        yi[i]-=bo[i];
    }

    for(int i=0;i<OUTPUT_LAYER_NEURE_NUM;i++){
        yo[i]=ffunc_Sigmoid(yi[i]);
    }

    E=Efunc(Do,yo,k);
    cout<<"--  Forward Conduction Over  --"<<endl<<endl;

    cout<<"--  Start Updating Weights  --"<<endl;
    UpdateWho(yo,who,Do,ho,partial_ihEO,k);
    UpdateWih(ho,wih,hi,inputVector,partial_ihEO,k);
    cout<<"--  Update Weight over  --"<<endl;
    cout<<"E = "<<E<<endl;
    return E;
}

int main(){
    double inputVector[INPUT_LAYER_NEURE_NUM][K];
    
    double Do[OUTPUT_LAYER_NEURE_NUM][K];   //目标值
    double bh[HIDDEN_LAYER_NEURE_NUM];  //隐含层阈值
    double bo[OUTPUT_LAYER_NEURE_NUM];  //输出层阈值
    double eps;   //计算精度
    double E=0;   //误差
    int M;      //最大学习次数
    int k=1;    //默认第k个输出样本
    
    
    double wih[HIDDEN_LAYER_NEURE_NUM][INPUT_LAYER_NEURE_NUM];
    double who[OUTPUT_LAYER_NEURE_NUM][HIDDEN_LAYER_NEURE_NUM];

    ReadFileInput(inputVector);
    ReadFileDo(Do);
    Initialization(wih,who,bh,bo,eps,M);
    int getM=M;
    while(M!=0){
        k=rand()%K; //随机选取第k组训练数据
        E=training(inputVector,Do,wih,who,bh,bo,k);
        M--;
        if(E<=eps){
            break;
        }
    }
    cout<<"--  Training over  --"<<endl;
    cout<<"Training Times: "<<getM-M<<endl;
    return 0;
}
