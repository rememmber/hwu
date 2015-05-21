//   File:          advanced_genetic_algorithm_supervisor.c
//   Description:   Supervisor code for genetic algorithm
//   Project:       Advanced exercises in Cyberbotics' Robot Curriculum
//   Author:        Yvan Bourquin - www.cyberbotics.com
//   Date:          January 6, 2010

//   Editor:        Boris Mocialov (bm4@hw.ac.uk)
//   Date:          20.05.2015

#include "genotype.h"
#include <webots/supervisor.h>
#include <webots/robot.h>
#include <webots/emitter.h>
#include <webots/receiver.h>
#include <webots/display.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if (defined _WIN32 || defined __WIN64)
    #include <windows.h>
#elif defined __APPLE__    
    #include <spawn.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif


#include <sys/stat.h>

#if (defined _WIN32 || defined __WIN64)
    static const char *GENOTYPE_FILE_NAME = "genes.txt"; //"/Users/master/Desktop/test_processes/fifofile";

    static const char *GENOTYPE_FITNESS_FILE_NAME = "genes_fitness.txt";

    static const char *BEST_GENOTYPE_FILE_NAME = "best_solution.txt"; //"/Users/master/Desktop/test_processes/best_solution";

    static const char *PYTHON_PATH = "C:\\Windows\\py.exe "; //used for WIN (for UNIX, path is specified inside the .py file)
#elif defined __APPLE__
    static const char *GENOTYPE_FILE_NAME = "genes.txt";

    static const char *GENOTYPE_FITNESS_FILE_NAME = "genes_fitness.txt";

    static const char *BEST_GENOTYPE_FILE_NAME = "best_solution.txt";
#else
    static const char *GENOTYPE_FILE_NAME = "genes.txt";

    static const char *GENOTYPE_FITNESS_FILE_NAME = "genes_fitness.txt";

    static const char *BEST_GENOTYPE_FILE_NAME = "best_solution.txt";
#endif

// must match the values in the advanced_genetic_algorithm.c code
static const int NUM_SENSORS = 10;
static const int NUM_hidden = 9;
static const int NUM_WHEELS  = 2;
#define GENOTYPE_SIZE (NUM_SENSORS + NUM_hidden + NUM_WHEELS) * (NUM_SENSORS + NUM_hidden + NUM_WHEELS)

//For peas
extern char **environ;

#if (defined _WIN32 || defined __WIN64)
    const char *python_prog = "..\\..\\..\\peas\\test\\line_following_webots.py"; //"C:\\Users\\ifxboris\\Desktop\\hwu\\peas\\test\\line_following_webots.py"; //"/Users/master/Desktop/peas/peas/peas/test/line_following_webots.py";
#elif defined __APPLE__
    const char *python_prog = "../../../peas/test/line_following_webots.py";
#else
    const char *python_prog = "";
#endif

// index access
enum { X, Y, Z };

static int time_step;
static WbDeviceTag emitter;   // to send genes to robot
static WbDeviceTag receiver;  // for receiving genes from slave
static WbDeviceTag display;   // to display the fitness evolution
static int display_width, display_height;

// for reading or setting the robot's position and orientation
static WbFieldRef robot_translation;
static WbFieldRef robot_rotation;
static double robot_trans0[3];  // a translation needs 3 doubles
static double robot_rot0[4];    // a rotation needs 4 doubles

// start with a demo until the user presses the 'O' key
// (change this if you want)
static bool demo = false;

int steps=0; //steps taken during one run (there is also another way to get this value)
double total_fitness; //holds information about total fitness during one run
const double *data_values; //holds data values received from the controller
//int obstacle_on_side = 0; //holds obstacle present on side counter
//double side_obstacle_sensor_val = 0.0; //holds temporary value for side obstacle sensor


int isEmpty(FILE *file)
{
    long savedOffset = ftell(file);
    fseek(file, 0, SEEK_END);
    
    if (ftell(file) == 0)
    {
        return 1;
    }
    
    fseek(file, savedOffset, SEEK_SET);
    return 0;
}

int file_exist (char *filename)
{
    struct stat   buffer;
    return (stat (filename, &buffer) == 0);
}

//check whether slave have sent values to the supervisor
void check_for_slaves_data(){
    if (wb_receiver_get_queue_length(receiver) > 0) {
        double wheel_speed1, wheel_speed2; //wheels speed
        double ground1, ground2;//, ground3; //ground sensors
        
        ground1 = ground2 = 0; // = ground3 = 0; //initialise ground sensors
        wheel_speed1 = wheel_speed2 = 0; //initialise speed values
        
        data_values = wb_receiver_get_data(receiver); //copy data values over from controller (should use asser before)
        double max_right_sensor_value = 0.0; //declare and initialise variable to hold maximum sensor activation on the right side of the robot
        double max_left_sensor_value = 0.0; //declare and initialise variable to hold maximum sensor activation on the left side of the robot
        
        int i;
        double max_sensor_value = 0.0;
        for (i = 0; i <= 11; i++){  //[0..11] -> 11 sensors + 2 wheel speeds
            if(i>=10 && i<=11){ //copy wheel speeds
                if(i == 10){
                    wheel_speed1 = *(data_values+i);
                }else if(i == 11){
                    wheel_speed2 = *(data_values+i);
                }
            }else if(i<=9){
                if(i < 8){  // proximity sensor values [0..8]
                    if(i == 4 || i == 5 || i == 6){ //right-side sensors
                        if(max_right_sensor_value <= *(data_values+i)){
                            max_right_sensor_value = *(data_values+i);
                        }
                    }
                    
                    if(i == 1 || i == 2 || i == 3){ //left-side sensors
                        if(max_left_sensor_value <= *(data_values+i)){
                            max_left_sensor_value = *(data_values+i);
                        }
                    }
                    
                    if(max_sensor_value <= *(data_values+i)){
                        max_sensor_value = *(data_values+i); //maximum proximity sensor activation
                    }
                }
                
                //3 ground sensors
                if(i == 8){
                    ground1 = *(data_values+i);
                }else if(i == 9){
                    ground2 = *(data_values+i);
                }
                //else if(i == 10){
                //    ground3 = *(data_values+i);
                //}
            }
        }
        
        int on_line1 = (ground1 >= 250 && ground1 <= 400 && max_sensor_value < 500);
        int on_line2 = (ground2 >= 250 && ground2 <= 400 && max_sensor_value < 500);
        double on_line_dlt = (on_line1 + on_line2) / (double)2;
        
        // Follow wall
        double currentFitness1 = (double)0;
        if (wheel_speed1 < (double)300 && wheel_speed2 < (double)300) currentFitness1 -= (double)1;      // Punish slow speed
        if (max_right_sensor_value > (double)3000 || max_left_sensor_value > (double)3000) currentFitness1 += (double)1;         // Reward max IR activation of side sensors
        if (wheel_speed1 == 0 && wheel_speed2 == 0) currentFitness1 -= (double)1;        // Penalise standing still
        if ((abs(wheel_speed1) - wheel_speed2) >= (double)50) currentFitness1 -= (double)2;// Penalise oscillatory movement
        total_fitness += currentFitness1;
        
        
        // Avoid obstacles:
        double currentFitness0 = (double)0;
        if (max_sensor_value > (double)3000) currentFitness0 -= (double)1;   // Punish hitting obstacles
        if ((abs(wheel_speed1) - wheel_speed2) >= (double)50) currentFitness0 -= (double)1;    // Punish oscillatory movements
        if (wheel_speed1 == (double)0 && wheel_speed2 == (double)0) currentFitness0 -= (double)1;    // Punish standing still
        if (wheel_speed1 > (double)300 && wheel_speed2 > (double)300) currentFitness0 += (double)2; // Reward fast speed
        total_fitness += currentFitness0;
        
        // Follow black line
        double currentFitness2 = (double)0;
        if (ground1 < (double)400 || ground2 < (double)400 && on_line_dlt != 0.0)
            currentFitness2 += (double)1; // Reward detection of black line
        if (wheel_speed1 < (double)200 && wheel_speed2 < (double)200) currentFitness2 -= (double)2;  // Punish slow speed
        if (on_line_dlt == 0) currentFitness2 -= (double)1;                        // Punish detection of white line
        total_fitness += currentFitness2;
        
        //  double max_side_sensor = max_left_sensor_value > max_right_sensor_value ? max_left_sensor_value : max_right_sensor_value;
        //
        //  int on_line1 = (ground1 >= 250 && ground1 <= 400 && max_sensor_value < 500);
        //  int on_line2 = (ground2 >= 250 && ground2 <= 400 && max_sensor_value < 500);
        //  //int on_line3 = (ground3 >= 250 && ground3 <= 400 && max_sensor_value < 500);
        //
        // int on_white1 = (ground1 > 400 && ground1 < 500);
        // //int on_white2 = (ground2 > 700);
        // int on_white3 = (ground2 > 700);
        //
        // double on_line_dlt = (on_line1 + on_line2) / (double)2;
        //
        // double on_white_dlt = on_white1 && on_white3;
        //
        //if(on_white_dlt && data_values[2] > 300 && data_values[2] < 1000){
        // printf("on_white\n");
        // total_fitness += 50;
        //}
        //
        // double dlt_v = (double)abs(((wheel_speed1 + (double)1000) - (wheel_speed2 + (double)1000)))/(double)2000;
        //
        // double i_ = (max_sensor_value - (on_white_dlt && data_values[2] > 300 && data_values[2] < 1000) ? data_values[2]/(double)4096 : 0) / (double)4096;
        //
        // double V = (wheel_speed1 + wheel_speed2)/(double)2000;
        //
        //int going_around_obstacle = (i2_ > (double)0.1 && on_white_dlt > (double)0.4) || (on_line_dlt > (double)0.4 && i_ > (double)0.1);
        //
        // double nolfi = ((V * ((double)1 - (double)sqrt(dlt_v)) * ((double)1 - i_)) + on_line_dlt);// + (on_white_dlt && data_values[2] > 300 && data_values[2] < 1000) ? data_values[2]/(double)15096 : 0);
        // double _nolfi = ((V * ((double)1 - (double)sqrt(dlt_v)) * ((double)1 - i_)) - on_line_dlt);// - (on_white_dlt && data_values[2] > 300 && data_values[2] < 1000) ? data_values[2]/(double)15096 : 0);
        //
        // total_fitness += (nolfi - (double)abs(_nolfi));
        // if(max_sensor_value < 500 && on_white_dlt && data_values[2] > 200 && data_values[2] < 300){
        //   obstacle_on_side++;
        //   side_obstacle_sensor_val += (data_values[2])/(double)4096;
        // }
        //
        // if(obstacle_on_side > 10 && max_sensor_value < 500 && on_white_dlt && data_values[2] > 200 && data_values[2] < 300){
        //   total_fitness -= side_obstacle_sensor_val;
        //   side_obstacle_sensor_val = 0.0;
        //   obstacle_on_side = 0;
        // }else{
        //   total_fitness += side_obstacle_sensor_val;
        //   side_obstacle_sensor_val = 0.0;
        //   obstacle_on_side = 0;
        // }
        
        steps++;
        
        // prepare for receiving next slaves data packet
        wb_receiver_next_packet(receiver);
    }
}

void draw_scaled_line(int generation, double y1, double y2) {
    //const double XSCALE = (double)display_width / NUM_GENERATIONS;
    //const double YSCALE = 10.0;
    //wb_display_draw_line(display, (generation - 0.5) * XSCALE, display_height - y1 * YSCALE,
    //(generation + 0.5) * XSCALE, display_height - y2 * YSCALE);
}

// plot best and average fitness
void plot_fitness(int generation, double best_fitness, double average_fitness) {
    static double prev_best_fitness = 0.0;
    static double prev_average_fitness = 0.0;
    if (generation > 0) {
        wb_display_set_color(display, 0xff0000); // red
        draw_scaled_line(generation, prev_best_fitness, best_fitness);
        
        wb_display_set_color(display, 0x00ff00); // green
        draw_scaled_line(generation, prev_average_fitness, average_fitness);
    }
    
    prev_best_fitness = best_fitness;
    prev_average_fitness = average_fitness;
}

void plot_trajectory(){
    if (demo){
        memcpy(robot_trans0, wb_supervisor_field_get_sf_vec3f(robot_translation), sizeof(robot_trans0));
        wb_display_draw_pixel(display, display_width/2-robot_trans0[2]*-display_width, display_height/2-robot_trans0[0]*display_height);
    }
}

// run the robot simulation for the specified number of seconds
void run_seconds(double seconds) {
    int i, n = 1000.0 * seconds / time_step;
    for (i = 0; i < n; i++) {
        check_for_slaves_data();
        //plot_trajectory();
        
        //if (demo && wb_robot_keyboard_get_key() == 'O') {
        //    demo = false;
        //    return; // interrupt demo and start GA optimization
        //}
        
        wb_robot_step(time_step);
    }
}

// compute fitness as the euclidian distance that the load was pushed
double measure_fitness() {
    printf("fitness of a run: %f\n", total_fitness/(double)steps);
    return total_fitness/(double)steps;
}

// evaluate one genotype at a time
void evaluate_genotype(Genotype genotype) {
    steps = 0; //reset steps for the next evaluation
    
    double data_values[genotype_get_size()]; //set data size for array to receive data from controller
    int i;
    for (i=0; i<genotype_get_size(); i++){
        data_values[i] = genotype_get_genes(genotype)[i];
    }
    
    wb_emitter_send(emitter, data_values, genotype_get_size()*sizeof(double));
    
    free(genotype);
    
    // reset robot and load position
    wb_supervisor_field_set_sf_vec3f(robot_translation, robot_trans0);
    wb_supervisor_field_set_sf_rotation(robot_rotation, robot_rot0);
    
    // evaluation genotype during one minute
    run_seconds(240.0);
    
    // measure fitness
    double fitness = measure_fitness();
    
    char output_fitness[1*sizeof(double)];
    snprintf(output_fitness,sizeof(output_fitness),"%f",fitness);
    
    FILE *file;
    file = fopen(GENOTYPE_FITNESS_FILE_NAME,"w"); //fopen("/Users/master/Desktop/test_processes/fifofile_fitness","w");
    fprintf(file, output_fitness);
    fclose(file);
}

void run_optimization() {
    int times = 0;
    int redo = 1;
    while(redo){
        FILE *fp = fopen(GENOTYPE_FILE_NAME, "rb");
        if ( fp == NULL ){
            fclose(fp);
            sleep(1);
            continue;
            run_optimization();
        }
        
        while ( !feof (fp) )
        {
            redo = 0;
            if(isEmpty(fp)){
            fclose(fp);
                continue;
            }else{
                times++;
                
                if(times <= 1){
                    genotype_set_size(GENOTYPE_SIZE);
                    Genotype genotype = genotype_create();
                    genotype_fread(genotype, fp);
                    fclose(fp);
                    //removing file instead
                    remove(GENOTYPE_FILE_NAME);
                    
                    total_fitness = 0.0;
                    evaluate_genotype(genotype);
                    break;
                }else{
                    break;
                }
            }
            
        }
    }
    
    if(file_exist(GENOTYPE_FILE_NAME) || file_exist(GENOTYPE_FITNESS_FILE_NAME)){
        run_optimization();
    }
    
    wb_robot_cleanup();
    exit(0);
}

// show demo of the fittest individual
void run_demo() {
    FILE *fp = fopen(BEST_GENOTYPE_FILE_NAME, "r");
    
    genotype_set_size(GENOTYPE_SIZE);
    
    Genotype genotype = genotype_create();
    genotype_fread(genotype, fp);
    
    
    while (demo){
        evaluate_genotype(genotype);
    }
}

int main(int argc, const char *argv[]) {
    // initialize Webots
    wb_robot_init();
    
    // get simulation step in milliseconds
    time_step = wb_robot_get_basic_time_step();
    
    // the emitter to send genotype to robot
    emitter = wb_robot_get_device("emitter");
    
    // find and enable receiver
    receiver = wb_robot_get_device("receiver");
    wb_receiver_enable(receiver, time_step);
    
    // to display the fitness evolution
    display = wb_robot_get_device("display");
    display_width = wb_display_get_width(display);
    display_height = wb_display_get_height(display);
    
    if(!demo)
    wb_display_draw_text(display, "fitness", 2, 2);
    else
        wb_display_draw_text(display, "Trajectory", 2, 2);
    
    // find robot node and store initial position and orientation
    WbNodeRef robot = wb_supervisor_node_get_from_def("EPUCK");
    robot_translation = wb_supervisor_node_get_field(robot, "translation");
    robot_rotation = wb_supervisor_node_get_field(robot, "rotation");
    memcpy(robot_trans0, wb_supervisor_field_get_sf_vec3f(robot_translation), sizeof(robot_trans0));
    memcpy(robot_rot0, wb_supervisor_field_get_sf_rotation(robot_rotation), sizeof(robot_rot0));
    
    if (demo)
        run_demo();
    
    int parentID = getpid();
    char str[1*sizeof(double)];
    sprintf(str, "%d", parentID);
    char* name_with_extension;
    name_with_extension = malloc(2+strlen(python_prog)+1*sizeof(int)+1+sizeof(GENOTYPE_FILE_NAME)+1); //malloc(strlen(python_prog)+strlen(" ")+strlen(" ")+strlen(GENOTYPE_FILE_NAME));
    strcpy(name_with_extension, python_prog);
    strcat(name_with_extension, " ");
    strcat(name_with_extension, str);
    strcat(name_with_extension, " ");
    strcat(name_with_extension, GENOTYPE_FILE_NAME);
    
    pid_t pid;
    char *argvv[] = {"sh", "-c", name_with_extension, NULL};
    int status;
    
    #if (defined _WIN32 || defined __WIN64)
        
        STARTUPINFO si;
        PROCESS_INFORMATION pi;
    
        ZeroMemory( &si, sizeof(si) );
        si.cb = sizeof(si);
        ZeroMemory( &pi, sizeof(pi) );
    
        char str2[strlen(PYTHON_PATH)+strlen(name_with_extension)];
        strcpy(str2, PYTHON_PATH);
        strcat(str2, name_with_extension);
    
        if (!CreateProcess(NULL, str2, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi))
        {
            printf( "CreateProcess failed (%d).\n", GetLastError() );
        }
    
        //WaitForSingleObject( pi.hProcess, INFINITE );
    
        CloseHandle(pi.hThread);
    
    
    #elif defined __APPLE__
        status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argvv, environ);
    #else
        
    #endif
    
    // run GA optimization
    run_optimization();
    
    // cleanup Webots
    wb_robot_cleanup();
    return 0;  // ignored
}