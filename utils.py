import sys
from copy import deepcopy

import torch
import random
from numpy.random import RandomState
import os
import numpy as np
from pythomata import SimpleDFA, SymbolicAutomaton
import math
import torch.nn.functional as F
from losses import final_states_loss, not_final_states_loss

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def eval_accuracy(classifier, deepAutoma, test_loader, automa_implementation = 'logic_circuit', verbose = False):
    total = 0
    correct = 0
    classifier.eval()
    if automa_implementation == "dfa":
        temp= 0.00001
    else:
        temp=1

    with torch.no_grad():

        for (batch_image_dataset, batch_acceptance) in test_loader:
            batch_image_dataset = batch_image_dataset.squeeze(0).to(device)
            batch_acceptance = batch_acceptance.squeeze(0).to(device)
            batch, length, channels, pixels1, pixels2 = batch_image_dataset.size()

            sym_sequence = classifier(batch_image_dataset.view(-1, channels, pixels1, pixels2))
            if automa_implementation == "fuzzy_DFA":
                prediction = []
                sym_sequence = sym_sequence.view(batch, length, -1)
                for i in range(batch):
                    target = batch_acceptance[i]
                    last_state = torch.argmax(deepAutoma(sym_sequence[i])).item()
                    prediction.append(int(last_state in deepAutoma.final_states))

                prediction = torch.LongTensor(prediction)

            elif automa_implementation == "logic_circuit" or automa_implementation == "dfa":
                _, prediction = deepAutoma(sym_sequence.view(batch, length, -1))
                prediction = prediction[:,-1,:]
                prediction =prediction.view(-1, classifier.classes).to(device)
                prediction = torch.argmax(prediction, dim=-1)
            else:
                prediction = deepAutoma(sym_sequence.view(batch, length, -1))
                prediction =prediction.view(-1, classifier.classes)
                prediction = torch.argmax(prediction, dim=-1)


            if verbose:
                print("prediction:-------")
                print(prediction)
                print("target:---------")
                print(batch_acceptance)

            correct += torch.sum(prediction.to(device) == batch_acceptance).item()
            total += prediction.size()[0]

    return 100 * correct / total

def eval_accuracy_DFA(clasifier, dfa, X, y):
    total = 0
    correct = 0

    print(dfa.__dict__)
    print(dfa._alphabet.__dict__)
    print(dfa.accepts('000'))
    #TODO
    assert False

def eval_acceptance(classifier, automa, final_states, dfa, alphabet, dataset, automa_implementation='dfa', mutually_exc_sym=True):
    #automa implementation =
    #   - 'dfa' use the perfect dfa given
    #   - 'lstm' use the lstm model
    #   - 'logic_circuit' use the fuzzy automaton
    total = 0
    correct = 0
    test_loss = 0
    classifier.eval()

    with torch.no_grad():
        for i in range(len(dataset[0])):
            images = dataset[0][i].to(device)
            label = dataset[1][i]
            # primo modo usando la lstm o l'automa continuo
            if automa_implementation == 'lstm':
                accepted = automa(classifier(images))
                accepted = accepted[-1]

                output = torch.argmax(accepted).item()


            #secondo modo usando l'automa
            elif automa_implementation == 'dfa':
                pred_labels = classifier(images)
                if mutually_exc_sym:
                    pred_labels = pred_labels.data.max(1, keepdim=False)[1]

                    trace = []
                    for p_l in pred_labels:
                        truth_v = {}
                        for symbol in alphabet:
                            truth_v[symbol] = False

                        truth_v[alphabet[p_l.item()]] = True
                        trace.append(truth_v)
                else:
                    trace = []

                    for pred in pred_labels:
                        truth_v = {}
                        for i, symbol in enumerate(alphabet):
                            if pred[i] > 0.5:
                                truth_v[symbol] = True
                            else:
                                truth_v[symbol] = False
                        trace.append(truth_v)

                output = int(dfa.accepts(trace))

            #terzo modo: usando il circuito logico continuo
            elif automa_implementation == 'logic_circuit':
                sym = classifier(images)

                last_state = automa(sym)
                last_state = torch.argmax(last_state).item()

                output = int(last_state in final_states)


            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)

            total += 1


            correct += int(output==label)


        test_accuracy = 100. * correct/(float)(total)

    return test_accuracy

def eval_image_classification_from_traces(traces_images, traces_labels, classifier, mutually_exclusive):
    total = 0
    correct = 0
    classifier.eval()


    with torch.no_grad():
        for i in range(len(traces_labels)) :
            t_sym = traces_labels[i].to(device)
            t_img = traces_images[i].to(device)

            pred_sym = classifier(t_img)

            if  not mutually_exclusive:

                y1 = torch.ones(t_sym.size()).to(device)
                y2 = torch.zeros(t_sym.size()).to(device)

                output_sym = pred_sym.where(pred_sym <= 0.5, y1)
                output_sym = output_sym.where(pred_sym > 0.5, y2)

                correct += torch.sum(output_sym == t_sym).item()
                total += torch.numel(pred_sym)

            else:
                output_sym = pred_sym.data.max(1, keepdim=True)[1]

                t_sym = t_sym.data.max(1, keepdim=True)[1]

                correct += torch.sum(output_sym == t_sym).item()
                total += t_sym.size()[0]

    accuracy = 100. * correct / (float)(total)
    return accuracy

def dot2pythomata(dot_file_name, action_alphabet):

        fake_action = "(~" + action_alphabet[0]
        for sym in action_alphabet[1:]:
            fake_action += " & ~" + sym
        fake_action += ") | (" + action_alphabet[0]
        for sym in action_alphabet[1:]:
            fake_action += " & " + sym
        fake_action += ")"

        file1 = open(dot_file_name, 'r')
        Lines = file1.readlines()

        count = 0
        states = set()

        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                action = line.strip().split('"')[1]
                states.add(line.strip().split(" ")[0])
            else:
                if "doublecircle" in line.strip():
                    final_states = line.strip().split(';')[1:-1]

        automaton = SymbolicAutomaton()
        state_dict = dict()
        state_dict['0'] = 0
        for state in states:
            if state == '0':
                continue
            state_dict[state] = automaton.create_state()

        final_state_list = []
        for state in final_states:
            state = int(state)
            state = str(state)
            final_state_list.append(state)

        for state in final_state_list:
            automaton.set_accepting_state(state_dict[state], True)

        count = 0
        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                action = line.strip().split('"')[1]
                action_label = action
                for sym in action_alphabet:
                    if sym != action:
                        action_label += " & ~" + sym
                init_state = line.strip().split(" ")[0]
                final_state = line.strip().split(" ")[2]
                automaton.add_transition((state_dict[init_state], action_label, state_dict[final_state]))
                automaton.add_transition((state_dict[init_state], fake_action, state_dict[init_state]))

        automaton.set_initial_state(state_dict['0'])

        return automaton


def transacc2pythomata(trans, acc, action_alphabet, initial_state=0):
    accepting_states = set()
    for i in range(len(acc)):
        if acc[i]:
            accepting_states.add(i)

    automaton = SimpleDFA.from_transitions(initial_state, accepting_states, trans)

    return automaton


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    """
    Gumbel-Softmax sampling function.
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def count_states_from_dot(dot_file_path):
    """
    Extract the number of states from a DOT file by counting lines between the first
    and second occurrence of 'fake'.

    Args:
        dot_file_path (str): Path to the DOT file.

    Returns:
        int: Number of states in the automaton.
    """
    try:
        with open(dot_file_path, 'r') as file:
            lines = file.readlines()

        # Find the indices of the lines containing 'fake'
        fake_indices = [i for i, line in enumerate(lines) if 'fake' in line]

        if len(fake_indices) < 2:
            raise ValueError("The DOT file does not contain at least two 'fake' entries.")

        # Extract lines between the two occurrences of 'fake'
        start_index = fake_indices[0] + 1
        end_index = fake_indices[1]

        # Count lines with actual states
        state_lines = [line for line in lines[start_index:end_index] if line.strip() and not line.strip().startswith('//')]

        return len(state_lines)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {dot_file_path} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the DOT file: {e}")


def count_symbols_from_dot(dot_file_path):
    """
    Extract the set of labels from a DOT file.

    Args:
        dot_file_path (str): Path to the DOT file.

    Returns:
        set: A set of labels in the graph.
    """
    try:
        with open(dot_file_path, 'r') as file:
            lines = file.readlines()

        labels = set()

        for line in lines:
            line = line.strip()
            if 'label=' in line:
                # Extract the label value
                start = line.find('label=') + len('label=')
                end = line.find(']', start)
                if end == -1:
                    end = len(line)
                label = line[start:end].strip().strip('"')

                # Add the label to the set
                if label.isdigit():
                    labels.add(int(label))

        return len(labels)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {dot_file_path} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the DOT file: {e}")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, min_loss= 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = min_loss
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset contatore se migliora
        else:
            self.counter += 1  # Conta le epoche senza miglioramento

        return (self.counter >= self.patience) or (val_loss < self.min_loss)  # Stop se superiamo la pazienza