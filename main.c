#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//Change input size accordingly to your needs

#define wdim 50
#define input_size 8
#define output_size 8
#define hidden_layers_number 8
#define hidden_layers_size 8
#define learning_rate 10

//Output file to save the layers
FILE *layersW;

//Input file from which to load the layers
FILE *loadW;

//Input file from which to load the sample data
FILE *loadS;

typedef struct _neuron{
    float sum;
    float activation;
}neuron;

typedef struct layer{
    float *weights;
    neuron *signals;
    int rows;
    int cols;
}layer;

typedef struct info{
    int ntot;
    int nh;
    int inp;
    int hid;
    int out;
}info;

typedef struct s_signal{
    int size;
    float *values;
}s_signal;

typedef struct network{
    struct info details;
    struct layer * section;
}network;

//Initializes the network with the amount of layers specified above
void initnetwork(network *N);

//Saves the net (weights) on a file
void saveNet(network *N);

//Propagates the signal forward from the input layer to the output layer
void forwardSignal(network *N, s_signal *input, s_signal *output);

//Load the weights of a network from a file
void loadnetwork(network *N,FILE *f);

//Prints the layers on the console (used for debug)
void printLayers(network *N);

//Trains the net to learn based on the samples
void trainNet(network *N, s_signal *output, s_signal *sample_data,int sample_size);

//Load sample data from file
void loadSamples(s_signal *sample_data, FILE *f);

//Sigmoid funcion, written inline to speed the process
static inline float sigmoid(float x);

//Derivate of the sigmoid funciont, written inline to speed the process
static inline float d_sigmoid(float x);


int main(void){
    network *brain=malloc(sizeof(brain));
    s_signal *input=malloc(sizeof(s_signal));
    s_signal *output=malloc(sizeof(s_signal));
    s_signal *samples=malloc(sizeof(s_signal));
    brain->details.ntot=4;
    brain->section=malloc(brain->details.ntot*sizeof(layer));
    if(loadW=fopen("weights.txt","r")){
        printf("\nCaricamento...\n");
        loadnetwork(brain,loadW);
    }else{
        printf("\nInizializzazione nuova rete...\n\n");
        initnetwork(brain,8,12,2,8);
        saveNet(brain);
    }
    input->size=input_size;
    output->size=output_size;
    input->values=malloc(input.size*sizeof(float));
    output->values=malloc(output.size*sizeof(float));
    srand(time(NULL));
    for(int i=0;i<input->size;i++){
        input->values[i]=(float)rand()/RAND_MAX*0.98;
    }
    if(loadS=fopen("sample_data.txt","r")){
        loadSamples(samples,loadS);
    }
    forwardSignal(brain,input,output);
    printf("\n\nINPUT VALUES:: ");
    for(int i=0;i<input->size;i++){
        printf("%.3f ",input->values[i]);
    }
    printf("\n\nOUTPUT VALUES:: ");
    for(int i=0;i<output->size;i++){
        printf("%.3f ",output->values[i]);
    }
    printf("\n\n");
    printLayers(brain);
    return 0;
}

void printLayers(network *N){
    for(int k=0;k<N->details.ntot;k++){
        printf("Layer %d ::\n",k);
        for(int i=0;i<N->section[k].rows;i++){
            for(int j=0;j<N->section[k].cols;j++){
                printf("%.3f ",N->section[k].weights[i*N->section[k].cols+j]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void loadnetwork(network *N,FILE *f){
    fscanf(f,"%d %d %d %d %d",&N->details.ntot,&N->details.nh,&N->details.inp,&N->details.hid,&N->details.out);
    int k=0;
    int dim;
    int inp=N->details.inp;
    int hidden=N->details.hid;
    int out=N->details.out;
    int nh=N->details.nh;
    dim=inp*hidden;
    N->section[k].rows=inp;
    N->section[k].cols=hidden;
    N->section[k].weights=malloc(dim*sizeof(float));
    N->section[k].signals=malloc(hidden*sizeof(neuron));
    for(int i=0;i<dim;i++){
        fscanf(f,"%f",&N->section[k].weights[i]);
    }
    dim=hidden*hidden;
    for(k=1;k<=nh;k++){
        N->section[k].rows=hidden;
        N->section[k].cols=hidden;
        N->section[k].weights=malloc(dim*sizeof(float));
        N->section[k].signals=malloc(hidden*sizeof(neuron));
        for(int i=0;i<dim;i++){
            fscanf(f,"%f",&N->section[k].weights[i]);
        }
    }
    N->section[k].rows=hidden;
    N->section[k].cols=out;
    dim=hidden*out;
    N->section[k].weights=malloc(dim*sizeof(float));
    N->section[k].signals=malloc(out*sizeof(neuron));
    for(int i=0;i<dim;i++){
        fscanf(f,"%f",&N->section[k].weights[i]);
    }
}

void forwardSignal(network *N, s_signal *input, s_signal *output){
    int k=0;

    //From input layer to the first hidden layer
    for(int i=0;i<N->section[k].rows;i++){
        for(int j=0;j<N->section[k].cols;j++){
            N->section[k].signals[i].sum+=N->section[k].weights[i*N->section[k].cols+j]*input->values[j];
        }
        N->section[k].signals[i].activation=sigmoid(N->section[k].signals[i].sum);
    }
    
    //From the first hidden layer to the last layer
    for(int k=1;k<N->details.ntot;k++){
        for(int i=0;i<N->section[k].rows;i++){
            for(int j=0;j<N->section[k].cols;j++){
                N->section[k].signals[i].sum+=N->section[k].weights[i*N->section[k].cols+j]*N->section[k-1].signals[j].activation;
            }
            N->section[k].signals[i].activation=sigmoid(N->section[k].signals[i].sum);
        }
    }

    //Copy the result of the last layer to the output layer
    for(int i=0;i<N->section[N->details.ntot-1].cols;i++){
        output->values[i]=N->section[N->details.ntot-1].signals[i].activation;
    }
}

static inline float sigmoid(float x){
    return (float)1/1+exp(-x);
}

static inline float d_sigmoid(float x){
    return (float)sigmoid(x)*(1-sigmoid(x));
}

void saveNet(network *N){
    int dim=N->section[k].rows*N->section[k].cols;
    if(layersW=fopen("weights.txt","w+")){
        fprintf(layersW,"%d %d %d %d %d\n",N->details.ntot,N->details.nh,N->details.inp,N->details.hid,N->details.out);
        for(int k=0;k<N->details.ntot;k++){
            for(int i=0;i<dim;i++){
                fprintf(layersW,"%.3f ",N->section[k].weights[i]);
            }
        }
    }
}

void loadSamples(s_signal *sample_data, FILE *f){
    int n=0;
    puts("Loading sample data from file...\n");
    fscanf(f,"%d",sample_data->size);
    sample_data->values=malloc(sample_data->size*sizeof(float));
    for(int i=0;i<sample_data->size;i++){
        fscanf(f,"%f",sample_data->values[i]);
    }
    puts("Sample data loaded.\n");
}

void initnetwork(network *N){
    if(N->details.ntot<3){
        return;
    }
    int inp=input_size;
    int out=output_size;
    int nh=hidden_layers_number;
    int hidden=hidden_layers_size;
    int dim;
    N->details.inp=inp;
    N->details.out=out;
    N->details.hid=hidden;
    N->details.nh=nh;
    srand(time(NULL));
    int k=0;
    dim=hidden*inp;
    //Using matrix convenction to represent weights and matrix multiplication
    N->section[k].rows=hidden;
    N->section[k].cols=inp;
    N->section[k].weights=malloc(dim*sizeof(float));
    N->section[k].signals=malloc(hidden*sizeof(neuron));
    for(int i=0;i<dim;i++){
        N->section[k].weights[i]=(float)rand()/RAND_MAX*0.98;
    }
    dim=hidden*hidden;
    for(k=1;k<=nh;k++){
        N->section[k].rows=hidden;
        N->section[k].cols=hidden;
        N->section[k].weights=malloc(dim*sizeof(float));
        N->section[k].signals=malloc(hidden*sizeof(neuron));
        for(int i=0;i<dim;i++){
            N->section[k].weights[i]=(float)rand()/RAND_MAX*0.98;
        }
    }
    dim=hidden*out;
    N->section[k].rows=hidden;
    N->section[k].cols=out;
    N->section[k].weights=malloc(dim*sizeof(float));
    N->section[k].signals=malloc(out*sizeof(neuron));
    for(int i=0;i<dim;i++){
        N->section[k].weights[i]=(float)rand()/RAND_MAX*0.98;
    }
}

void trainNet(network *N, s_signal *output, s_signal *sample_data){
    //Cost function: quadratic mean 1/2*(Yi-yi)^2 --> derivate: (Yi-yi)
    

}
