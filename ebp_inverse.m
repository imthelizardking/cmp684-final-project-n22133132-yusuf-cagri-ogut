clear all; close all; clc;
load('training_sine_400.mat'); % load training data
%%  Normalize
c_train = (out.c-(min(out.c)+max(out.c))/2)/((max(out.c)-min(out.c))/2);
output_training = c_train;
y_train = (out.y-(min(out.y)+max(out.y))/2)/((max(out.y)-min(out.y))/2);
input_training = y_train;
%%
NUM_SAMPLES = max(size(input_training)); % get number of inputs
NUM_EPOCHS = 30; ETA = .1; % set training parameters
sz_input = size(input_training); sz_output = size(output_training); % get number of training pairs
NUM_INP = sz_input(2); NUM_HID = 3; NUM_OUT = sz_output(2); % set input, hidden and output layer sizes
BIAS = -.1; % set bias
error_epochs = zeros(NUM_EPOCHS,1); % initialize epoch error vector
Phi = rand(NUM_HID+NUM_OUT, max(NUM_INP,NUM_HID+1)); % initialize weight matrix
output_estimated = zeros(NUM_SAMPLES,NUM_OUT); % save output estimations for printing later
for counter_epoch=1:NUM_EPOCHS
    for counter_sample=1:NUM_SAMPLES 
        Phi_delta = zeros(NUM_HID+NUM_OUT, max(NUM_INP,NUM_HID+1)); % reset weight update matrix
        input_vector = [input_training(counter_sample,:)';BIAS]; % set input vector for current training pair
        % FORWARD PASS:
        S1 = Phi(1:NUM_HID,1:NUM_INP+1)*input_vector;
        O1 = tanh(S1);
        S2 = Phi(NUM_HID+1:end,1:NUM_HID+1)*[O1;BIAS];
        O2 = S2;
        output_estimated(counter_sample,:) = O2';
        error = output_training(counter_sample,:)' - O2;        
        % BACKWARD PASS:
        % W1 update:
        for counter_outputs=1:NUM_OUT
            Phi_delta(NUM_HID+1:end,1:NUM_HID+1) ...
                = Phi_delta(NUM_HID+1:end,1:NUM_HID+1) + ...
                  ETA * error(counter_outputs) * ...
                  [O1;BIAS]';
        end
        % W0 update:
        input_vector_repeated = [];
        for counter_repeat=1:NUM_HID
            input_vector_repeated = [input_vector_repeated; input_vector'];
        end
        for counter_outputs=1:NUM_OUT
            Phi_delta(1:NUM_HID,1:NUM_INP+1)...
                = Phi_delta(1:NUM_HID,1:NUM_INP+1) +...
                  ETA * error(counter_outputs) * ...
                  Phi(NUM_HID+counter_outputs,:) * ...
                  (1 - [O1;BIAS].^2) * ...
                  input_vector_repeated;
        end
        Phi = Phi + Phi_delta; % update weights
        error_epochs(counter_epoch) = error_epochs(counter_epoch) + sum(error); % calculate epoch error
    end
    error_epochs(counter_epoch)
end
%% Print and plot results
[input_training output_training output_estimated];
Phi
plot(error_epochs,'DisplayName','Epoch Errors'); grid minor; legend;