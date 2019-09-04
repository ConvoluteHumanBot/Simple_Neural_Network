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

FILE *layersW;
FILE *loadW;

typedef struct layer{
    float **weights;
    float *signals;
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

void initnetwork(network *N);
void saveNet(network *N);
void forwardSignal(network *N, s_signal *input, s_signal *output);
void loadnetwork(network *N,FILE *f);
void printLayers(network *N);
void trainNet(network *N);
static inline float sigmoid(float x);
/*
input layers: 8 squares around, health
actions take 1 health away from total, if it is 0 the bot dies and game ends.
output actions: move in 1 of 8 directions, attack, eat
*/

/*
Rules of the game: square world
*/

int main(void){
    network *brain=malloc(sizeof(brain));
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
    s_signal *input=malloc(sizeof(s_signal));
    s_signal *output=malloc(sizeof(s_signal));
    input->size=input_size;
    output->size=output_size;
    input->values=malloc(input.size*sizeof(float));
    output->values=malloc(output.size*sizeof(float));
    srand(time(NULL));
    for(int i=0;i<input->size;i++){
        input->values[i]=(float)rand()/RAND_MAX*0.98;
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
                printf("%.3f ",N->section[k].weights[i][j]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void loadnetwork(network *N,FILE *f){
    fscanf(f,"%d %d %d %d %d",&N->details.ntot,&N->details.nh,&N->details.inp,&N->details.hid,&N->details.out);
    int k=0;
    int inp=N->details.inp;
    int hidden=N->details.hid;
    int out=N->details.out;
    int nh=N->details.nh;
    N->section[k].rows=inp;
    N->section[k].cols=hidden;
    N->section[k].weights=malloc(inp*sizeof(float*));
    N->section[k].signals=malloc(hidden*sizeof(float));
    for(int i=0;i<inp;i++){
        N->section[k].weights[i]=malloc(hidden*sizeof(float));
        for(int j=0;j<hidden;j++){
            fscanf(f,"%f",&N->section[k].weights[i][j]);
        }
    }
    for(k=1;k<=nh;k++){
        N->section[k].rows=hidden;
        N->section[k].cols=hidden;
        N->section[k].weights=malloc(hidden*sizeof(float*));
        N->section[k].signals=malloc(hidden*sizeof(float));
        for(int i=0;i<hidden;i++){
            N->section[k].weights[i]=malloc(hidden*sizeof(float));
            for(int j=0;j<hidden;j++){
                fscanf(f,"%f",&N->section[k].weights[i][j]);
            }
        }
    }
    N->section[k].rows=hidden;
    N->section[k].cols=out;
    N->section[k].weights=malloc(hidden*sizeof(float*));
    N->section[k].signals=malloc(out*sizeof(float));
    for(int i=0;i<hidden;i++){
        N->section[k].weights[i]=malloc(out*sizeof(float));
        for(int j=0;j<out;j++){
            fscanf(f,"%f",&N->section[k].weights[i][j]);
        }
    }
}

void forwardSignal(network *N, s_signal *input, s_signal *output){
    int k=0;
    //Calculate the sum for every signal
    for(int i=0;i<N->section[k].rows;i++){
        for(int j=0;j<N->section[k].cols;j++){
            N->section[k].signals[j]+=N->section[k].weights[i][j]*input->values[i];
        }
    }
    //Compute the sigmoid funciont on every signal to actually have a proper activation value (0,1)
    for(int j=0;j<N->section[k].cols;j++){
        N->section[k].signals[j]=sigmoid(N->section[k].signals[j]);
    }
    //This for loop does the same thing over the hidden layers
    for(int k=1;k<N->details.ntot;k++){
        for(int i=0;i<N->section[k].rows;i++){
            for(int j=0;j<N->section[k].cols;j++){
                N->section[k].signals[j]+=N->section[k].weights[i][j]*N->section[k-1].signals[i];
            }
        }
        for(int j=0;j<N->section[k].cols;j++){
            N->section[k].signals[j]=sigmoid(N->section[k].signals[j]);
        }
    }
    for(int i=0;i<N->section[N->details.ntot-1].cols;i++){
        output->values[i]=N->section[N->details.ntot-1].signals[i];
    }
}

static inline float sigmoid(float x){
    return (float)1/1+exp(-x);
}

void saveNet(network *N){
    if(layersW=fopen("weights.txt","w+")){
        fprintf(layersW,"%d %d %d %d %d\n",N->details.ntot,N->details.nh,N->details.inp,N->details.hid,N->details.out);
        for(int k=0;k<N->details.ntot;k++){
            for(int i=0;i<N->section[k].rows;i++){
                for(int j=0;j<N->section[k].cols;j++){
                    fprintf(layersW,"%.3f ",N->section[k].weights[i][j]);
                }
            }
        }
    }
}

void initnetwork(network *N){
    if(N->details.ntot<3){
        return;
    }
    int inp=input_size;
    int out=output_size;
    int nh=hidden_layers_number;
    int hidden=hidden_layers_size;
    N->details.inp=inp;
    N->details.out=out;
    N->details.hid=hidden;
    N->details.nh=nh;
    srand(time(NULL));
    int k=0;
    N->section[k].rows=inp;
    N->section[k].cols=hidden;
    N->section[k].weights=malloc(inp*sizeof(float*));
    N->section[k].signals=malloc(hidden*sizeof(float));
    for(int i=0;i<inp;i++){
        N->section[k].weights[i]=malloc(hidden*sizeof(float));
        for(int j=0;j<hidden;j++){
            N->section[k].weights[i][j]=(float)rand()/RAND_MAX*0.98;
        }
    }
    for(k=1;k<=nh;k++){
        N->section[k].rows=hidden;
        N->section[k].cols=hidden;
        N->section[k].weights=malloc(hidden*sizeof(float*));
        N->section[k].signals=malloc(hidden*sizeof(float));
        for(int i=0;i<hidden;i++){
            N->section[k].weights[i]=malloc(hidden*sizeof(float));
            for(int j=0;j<hidden;j++){
                N->section[k].weights[i][j]=(float)rand()/RAND_MAX*0.98;
            }
        }
    }
    N->section[k].rows=hidden;
    N->section[k].cols=out;
    N->section[k].weights=malloc(hidden*sizeof(float*));
    N->section[k].signals=malloc(out*sizeof(float));
    for(int i=0;i<hidden;i++){
        N->section[k].weights[i]=malloc(out*sizeof(float));
        for(int j=0;j<out;j++){
            N->section[k].weights[i][j]=(float)rand()/RAND_MAX*0.98;
        }
    }
}

void trainNet(network *N){

}
