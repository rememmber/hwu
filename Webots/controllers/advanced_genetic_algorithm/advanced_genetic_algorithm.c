// File:          advanced_genetic_algorithm.c
// Description:   Robot execution code for genetic algorithm
// Project:       Advanced exercises in Cyberbotics' Robot Curriculum
// Author:        Yvan Bourquin - www.cyberbotics.com
// Date:          January 6, 2010

#include <webots/robot.h>
#include <webots/differential_wheels.h>
#include <webots/receiver.h>
#include <webots/emitter.h>
#include <webots/distance_sensor.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_SENSORS 10
#define NUM_hidden 9
#define NUM_WHEELS 2
#define GENOTYPE_SIZE ((NUM_SENSORS + NUM_hidden + NUM_WHEELS) * (NUM_SENSORS + NUM_hidden + NUM_WHEELS))

// sensor to wheels multiplication matrix
// each each sensor has a weight for each wheel
double *matrix;
double temp_matrix[21][21];

//total sensor values during one run
double sensor_values[NUM_SENSORS];
double hidden_value[NUM_hidden];
double wheel_speed[NUM_WHEELS];

WbDeviceTag sensors[NUM_SENSORS];  // proximity sensors
WbDeviceTag receiver;              // for receiving genes from Supervisor
WbDeviceTag emitter;   // to send mssg to supervisor

// check if a new set of genes was sent by the Supervisor
// in this case start using these new genes immediately
void check_for_new_genes() {
    if (wb_receiver_get_queue_length(receiver) > 0) {
        memcpy(temp_matrix, wb_receiver_get_data(receiver), GENOTYPE_SIZE * sizeof(double));
        
        wb_receiver_next_packet(receiver);
    }
}

static double clip_value(double value, double min_max) {
    if (value > min_max)
        return min_max;
    else if (value < -min_max)
        return -min_max;
    
    return value;
}

void report_step_state_to_supervisor(){
    double* data_values = malloc((NUM_SENSORS + NUM_WHEELS) * sizeof(double));
    
    int i;
    for(i=0;i<NUM_SENSORS; i++){
        data_values[i] = sensor_values[i];
    }
    
    for(i=0;i<NUM_WHEELS; i++){
        data_values[NUM_SENSORS+i] = wheel_speed[i];
    }
    
    // send message to supervisor
    wb_emitter_send(emitter, data_values, (NUM_SENSORS + NUM_WHEELS) * sizeof(double));
    
    free(data_values);
}

void sense_compute_and_actuate() {
    // read sensor values
    //double sensor_values[NUM_SENSORS];
    double max_sensor_value = 0.0;
    int i, j;
    for (i = 0; i < NUM_SENSORS; i++){
        sensor_values[i] = wb_distance_sensor_get_value(sensors[i]);
        if(i < 8){
            if (max_sensor_value < sensor_values[i]){
                max_sensor_value = sensor_values[i];
            }
        }
    }
    
    
    // compute actuation using Braitenberg's algorithm:
    // The speed of each wheel is computed by summing the value
    // of each sensor multiplied by the corresponding weight of the matrix.
    // By chance, in this case, this works without any scaling of the sensor values nor of the
    // wheels speed but this type of scaling may be necessary with a different problem
    //double wheel_speed[...] = { 0.0, 0.0 };
    
    if(max_sensor_value > (double)3000){
        wheel_speed[0] = 0.0;
        wheel_speed[1] = 0.0;
        
    }else{
        for (i = 0; i < NUM_hidden; i++){
            hidden_value[i] = 0.0;
            for (j = 0; j < NUM_SENSORS; j++){
                hidden_value[i] += temp_matrix[i+NUM_SENSORS][j] * sensor_values[j]/(double)4096;
            }
            hidden_value[i] = tanh(hidden_value[i]);
        }
        
        for (i = 0; i < NUM_WHEELS; i++){
            wheel_speed[i] = 0.0;
            for (j = 0; j < NUM_hidden; j++){
                wheel_speed[i] += temp_matrix[i+NUM_SENSORS+NUM_hidden][j+NUM_SENSORS] * hidden_value[j];
            }
            
            wheel_speed[i] = tanh(wheel_speed[i]) * (double)500.0 - sinh(wheel_speed[i]) * (double)100.0;
        }
    }
    // clip to e-puck max speed values to avoid warning [0 ... 1000]
    wheel_speed[0] = clip_value(wheel_speed[0], 1000.0);
    wheel_speed[1] = clip_value(wheel_speed[1], 1000.0);
    
    // actuate e-puck wheels
    wb_differential_wheels_set_speed(wheel_speed[0], wheel_speed[1]);
}

int main(int argc, const char *argv[]) {
    
    wb_robot_init();  // initialize Webots
    
    // find simulation step in milliseconds (WorldInfo.basicTimeStep)
    int time_step = wb_robot_get_basic_time_step();
    
    // find and enable proximity sensors
    char name[32];
    int i;
    for (i = 0; i < NUM_SENSORS; i++) {
        if(i<8){
            sprintf(name, "ps%d", i);
        }else{
            sprintf(name, "gs%d", (i-8));
        }
        
        sensors[i] = wb_robot_get_device(name);
        wb_distance_sensor_enable(sensors[i], time_step);
    }
    
    // find and enable receiver
    receiver = wb_robot_get_device("receiver2");
    wb_receiver_enable(receiver, time_step);
    
    // the emitter to send mssg to supervisor
    emitter = wb_robot_get_device("emitter");
    
    // initialize matrix to zero, hence the robot
    // wheels will initially be stopped
    
    // run until simulation is restarted
    while (wb_robot_step(time_step) != -1) {
        check_for_new_genes();
        sense_compute_and_actuate();
        
        if(wb_robot_step(time_step) == 0){
            report_step_state_to_supervisor();
        }
    }
    
    wb_robot_cleanup();  // cleanup Webots
    return 0;            // ignored
}