import numpy as np
import torch
import torch.nn as nn
import os
import time
import random
import pickle
import logging
import math
import torch.optim as optim
import pandas as pd
from collections import OrderedDict
import util

class NSMMPP(nn.Module):
    def __init__(self, label_size, hidden_size, args, prior=None):
        super(NSMMPP, self).__init__()
        self.args = args
        self.target = args.target
        self.device = args.device
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.num_eq = 7
        # add a special event label for initialization
        self.Emb = nn.Parameter(self.init_weight(torch.empty(hidden_size, label_size + 1, device=self.device)))
        self.W = nn.Parameter(self.init_weight(torch.empty(self.num_eq, hidden_size, hidden_size, device=self.device)))
        self.U = nn.Parameter(self.init_weight(torch.empty(self.num_eq, hidden_size, hidden_size, device=self.device)))
        self.d = nn.Parameter(torch.zeros(self.num_eq, hidden_size, device=self.device))
        self.w = nn.Parameter(self.init_weight(torch.empty(label_size, hidden_size, device=self.device)))
        self.log_s = nn.Parameter(torch.zeros(label_size, device=self.device))
        self.debug = False

    def init_weight(self, w):
        stdv = 1. / math.sqrt(w.size()[-1])
        w.uniform_(-stdv, stdv)
        return w

    def scaled_softplus(self, x):
        s = torch.exp(self.log_s)
        return s * self.softplus(x / s)

    # all zeros
    def init_hidden(self):
        c_t = torch.zeros(self.hidden_size, device=self.device)
        c_ = torch.zeros_like(c_t)
        h_t = torch.zeros_like(c_t)
        hidden = (c_t, h_t, c_, None, None, None)
        return hidden

    # only compute hidden variables
    def forward_one_step(self, label_prev, label, t_prev, t, hidden):
        c_t, h_t, c_, _, _, _ = hidden
        temp = self.W.matmul(label_prev) + self.U.matmul(h_t) + self.d
        i = self.sigmoid(temp[0, :])
        f = self.sigmoid(temp[1, :])
        z = self.tanh(temp[2, :])
        o = self.sigmoid(temp[3, :])
        i_ = self.sigmoid(temp[4, :])
        f_ = self.sigmoid(temp[5, :])
        delta = self.softplus(temp[6, :])
        c = f * c_t + i * z
        c_ = f_ * c_ + i_ * z
        c_t = c_ + (c - c_) * torch.exp(-delta * (t - t_prev))
        h_t = o * self.tanh(c_t)
        hidden = (c_t, h_t, c_, c, delta, o)
        return hidden

    def h_to_lambd(self, h):
        lambd_tilda = h.matmul(self.w.t())
        lambd = self.scaled_softplus(lambd_tilda)
        return lambd + 1e-9

    # compute NLL loss given a label_seq and a time_seq
    # sim_time_seq is simlulated times for computing integral
    def loglik(self, label_seq, time_seq, sim_time_seq, sim_time_idx, ignore_first):
        n = len(time_seq)
        # collect states right after each event
        # last event is EOS marker
        all_c = torch.zeros(n-1, self.hidden_size, device=self.device)
        all_c_ = torch.zeros_like(all_c)
        all_delta = torch.zeros_like(all_c)
        all_o = torch.zeros_like(all_c)
        all_h_t = torch.zeros_like(all_c)
        hidden = self.init_hidden()
        # BOS event is 0 at time 0
        label_prev = self.Emb[:, label_seq[0]].squeeze()
        t_prev = time_seq[0]
        for i in range(1,n):
            label = self.Emb[:, label_seq[i]].squeeze()
            t = time_seq[i]
            hidden = self.forward_one_step(label_prev, label, t_prev, t, hidden)
            _, all_h_t[i-1, :], all_c_[i-1, :], all_c[i-1, :], all_delta[i-1, :], all_o[i-1, :] = hidden
            label_prev = label
            t_prev = t
        if ignore_first:
            beg = 1
        else:
            beg = 0
        target = self.target
        h_t = all_h_t[beg:-1, :]
        if h_t.shape[0] > 0:
            lambd = self.h_to_lambd(h_t)
            term1 = (lambd[label_seq[(1+beg):-1] == target, target-1]).log().sum()
        else:
            term1 = 0
        c_sim = all_c_[sim_time_idx, :] + \
                (all_c[sim_time_idx, :] - all_c_[sim_time_idx, :]) * \
                torch.exp(-all_delta[sim_time_idx, :] * (sim_time_seq - time_seq[sim_time_idx])[:, None])
        h_sim = all_o[sim_time_idx, :] * self.tanh(c_sim)
        lambd_sim = self.h_to_lambd(h_sim)
        term2 = lambd_sim[:, target-1].mean() * (time_seq[-1] - time_seq[0])
        loglik = term1 - term2

        return -loglik, all_c, all_c_, all_delta, all_o, all_h_t, h_sim

    def forward(self, label_seq, time_seq, sim_time_seq, sim_time_idx, ignore_first):
        result = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx, ignore_first)
        return result[0].sum()

    def detect_outlier(self, label_seq, time_seq, sim_time_seq, sim_time_idx, sim_time_diffs, time_test, n_sample):
        with torch.no_grad():
            loglik, all_c, all_c_, all_delta, all_o, all_h_t, h_sim = self.loglik(
                    label_seq, time_seq, sim_time_seq, sim_time_idx, False)
            n = len(time_test)
            m = len(time_seq)
            score = torch.zeros(n-1, device=self.device)
            target = self.target
            j = 0
            ts = torch.zeros(n_sample, device=self.device)
            for i in range(n-1):
                t_beg = time_test[i]
                t_end = time_test[i+1]
                ts.uniform_(t_beg, t_end)
                # find the first event after t_beg
                while j < m and time_seq[j] <= t_beg:
                    j += 1
                assert(time_seq[j] > t_beg)
                assert(time_seq[j-1] <= t_beg)
                Lambd = 0
                k = j
                # calculate Lambda piecewise segmented by events
                while k < m and time_seq[k-1] <= t_end:
                    ts_in_range = ts[(ts > time_seq[k-1]) & (ts <= time_seq[k])]
                    if len(ts_in_range) > 0:
                        c = all_c[k-1,:]
                        c_ = all_c_[k-1,:]
                        delta = all_delta[k-1,:]
                        o = all_o[k-1,:]
                        c_ts = c_ + (c - c_) * torch.exp(-delta[None, :] * (ts_in_range[:, None] - time_seq[k-1]))
                        h_ts = o * self.tanh(c_ts)
                        lambd_all = self.h_to_lambd(h_ts)
                        lambd = lambd_all[:,target-1]
                        Lambd += lambd.sum() / n_sample
                    k += 1
                Lambd *= (t_end - t_beg)
                score[i] = Lambd
            lambd = self.h_to_lambd(all_h_t)[:-1]
            lambd_sim = self.h_to_lambd(h_sim)
            return score, -score, lambd, lambd_sim

    def detect_outlier_instant(self, label_seq, time_seq, sim_time_seq, sim_time_idx, sim_time_diffs, time_test):
        with torch.no_grad():
            loglik, all_c, all_c_, all_delta, all_o, all_h_t, h_sim = self.loglik(
                    label_seq, time_seq, sim_time_seq, sim_time_idx, False)
            n = len(time_test)
            m = len(time_seq)
            score = torch.zeros(n-1, device=self.device)
            target = self.target
            j = 0
            for i in range(n-1):
                t_end = time_test[i+1]
                if t_end == 0:
                    j = 1
                else:
                    # find the first event at/after t_end
                    while j < m and time_seq[j] < t_end:
                        j += 1
                    assert(time_seq[j] >= t_end)
                    # last event before t_end
                    assert(time_seq[j-1] < t_end)
                c = all_c[j-1,:]
                c_ = all_c_[j-1,:]
                delta = all_delta[j-1,:]
                o = all_o[j-1,:]
                c_ts = c_ + (c - c_) * torch.exp(-delta * (t_end - time_seq[j-1]))
                h_ts = o * self.tanh(c_ts)
                lambd_all = self.h_to_lambd(h_ts)
                lambd = lambd_all[target-1]
                score[i] = lambd
            lambd = self.h_to_lambd(all_h_t)[:-1]
            lambd_sim = self.h_to_lambd(h_sim)
            return score, -score, lambd, lambd_sim

class ModelManager:
    def __init__(self, train_set, val_set, test_set, save_path, args):
        self.args = args
        self.device = args.device
        self.target = args.target
        self.sim_time_diffs = None
        self.train_set, time_train, count_train = self.prepare(train_set, args.sample_multiplier)
        self.val_set, time_val, count_val = self.prepare(val_set, args.sample_multiplier)
        self.test_set, _, _ = self.prepare(test_set, args.sample_multiplier)
        self.horizon = (time_train+time_val)/(count_train+count_val)
        self.dt = 1
        self.model_path = save_path
        self.ignore_first = args.ignore_first

    def prepare(self, data_set, multiple, diff_sample_size=100, regular=False, step=None):
        if data_set is None:
            return None, None, None
        output = []
        total_time = 0
        total_count = 0
        for seq in data_set:
            label_seq = torch.tensor(
                np.concatenate(([0], seq['mark'], [0])),
                dtype=torch.long,
                device=self.device
            )
            time_seq = torch.tensor(
                np.concatenate(([seq['start']], seq['time'], [seq['stop']])),
                dtype=torch.float,
                device=self.device
            )
            n = len(time_seq)
            t0 = seq['start']
            tn = seq['stop']
            total_time += tn - t0
            total_count += (label_seq == 1).sum()
            if regular:
                sim_time_seq = torch.arange(t0, tn, step, device=self.device)
                sim_time_idx = torch.zeros_like(sim_time_seq, dtype=torch.long)
            else:
                sim_time_seq = time_seq.new_empty(n * multiple)
                sim_time_seq.uniform_(t0, tn)
                sim_time_idx = label_seq.new_zeros(n * multiple)
            for j in range(n - 1):
                sim_time_idx[(sim_time_seq > time_seq[j]) &
                             (sim_time_seq <= time_seq[j + 1])
                             ] = j
            if self.sim_time_diffs is None:
                temp = sim_time_seq.new_empty(diff_sample_size)
                temp.exponential_(1)
                self.sim_time_diffs, _ = torch.sort(temp)
            if seq['time_test'] is None:
                time_test = None
            else:
                time_test = torch.tensor(
                    seq['time_test'],
                    dtype=torch.float,
                    device=self.device)
            if seq['label_test'] is None:
                label_test = None
            else:
                label_test = torch.tensor(
                    seq['label_test'],
                    device=self.device,
                )
            item = {
                'id': seq['id'],
                'label_seq': label_seq,
                'time_seq': time_seq,
                'sim_time_seq': sim_time_seq,
                'sim_time_idx': sim_time_idx,
                'sim_time_diffs' : self.sim_time_diffs,
                'time_test': time_test,
                'label_test': label_test,
                'lambda_x': seq['lambda_x'],
                't_x': seq['t_x'],
            }
            output.append(item)
        return output, total_time, total_count

    def train_one_epoch(self, model, train_set, optimizer):
        log_interval = self.args.log_interval
        total_loss = 0
        total_num_seq = len(train_set)
        start_time = time.time()
        for seq_idx, item in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(item['label_seq'],
                         item['time_seq'],
                         item['sim_time_seq'],
                         item['sim_time_idx'],
                         False)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if seq_idx % log_interval == 0 and seq_idx > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logging.info('| {:5d}/{:5d} seqs | '
                             'ms/seq {:5.2f} | '
                             'loss {:5.2f} |'.format(
                                 seq_idx, total_num_seq,
                                 elapsed * 1000 / log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()

    def train(self, model, epochs=None, use_all_data=False, name="model.pt"):
        best_val_loss = None
        optimizer = optim.Adam(model.parameters(),
                               lr=self.args.lr)
        if epochs is None:
            early_stop = True
            epochs = self.args.epochs
        else:
            early_stop = False
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            if use_all_data:
                train_set = self.train_set + self.val_set
            else:
                train_set = self.train_set
            self.train_one_epoch(model, train_set, optimizer)
            val_loss = self.evaluate(model)
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | '
                         'valid loss {:f} |'.format(
                             epoch, (time.time() - epoch_start_time),
                             val_loss))
            logging.info('-' * 89)
            if best_val_loss is None or val_loss < best_val_loss:
                self.save_model(model, name)
                best_val_loss = val_loss
            elif early_stop:
                epochs = epoch - 1
                break
        if best_val_loss:
            self.load_model(model, name)
        return best_val_loss, epochs

    def evaluate(self, model):
        total_loss = 0
        for seq_idx, item in enumerate(self.val_set):
            loss = model(
                item['label_seq'],
                item['time_seq'],
                item['sim_time_seq'],
                item['sim_time_idx'],
                self.ignore_first
            )
            total_loss += loss.item()
        return total_loss

    def likelihood(self, model, dataset):
        total_loss = 0
        for seq_idx, item in enumerate(dataset):
            loss = model(
                item['label_seq'],
                item['time_seq'],
                item['sim_time_seq'],
                item['sim_time_idx'],
                self.ignore_first
            )
            total_loss += loss.item()
        return total_loss

    def detect_outlier(self, model, debug=False, instant=False):
        log_interval = self.args.log_interval
        results = []
        n = len(self.test_set)
        start_time = time.time()
        for seq_idx, item in enumerate(self.test_set):
            if instant:
                score_omiss, score_commiss, lambd, lambd_sim = model.detect_outlier_instant(
                    item['label_seq'],
                    item['time_seq'],
                    item['sim_time_seq'],
                    item['sim_time_idx'],
                    item['sim_time_diffs'],
                    item['time_test'],
                )
            else:
                score_omiss, score_commiss, lambd, lambd_sim = model.detect_outlier(
                    item['label_seq'],
                    item['time_seq'],
                    item['sim_time_seq'],
                    item['sim_time_idx'],
                    item['sim_time_diffs'],
                    item['time_test'],
                    n_sample=1000
                )
            df = pd.DataFrame(OrderedDict({
                'seq': seq_idx,
                'time': item['time_test'].numpy()[1:],
                'score_omiss': score_omiss.numpy(),
                'score_commiss': score_commiss.numpy(),
                'label': item['label_test'].numpy(),
            }))
            if item['id'] is None:
                df.insert(0, 'id', seq_idx+1)
            else:
                df.insert(0, 'id', item['id'])
            results.append(df)
            if (seq_idx+1) % log_interval == 0:
                logging.info(f'Finished detecting outliers in seq {seq_idx+1}/{n} in {(time.time()-start_time):.2f}s')
                start_time = time.time()
        results = pd.concat(results)
        return results

    def save_model(self, model, name):
        with open(os.path.join(self.model_path, name), 'wb') as f:
            torch.save(model.state_dict(), f)

    def load_model(self, model, name):
        with open(os.path.join(self.model_path, name), 'rb') as f:
            model.load_state_dict(torch.load(f))

class ContextDataLoader:
    def __init__(self, train_set, test_set, label_size, target=1):
        self.label_size = label_size
        self.target = target
        n = len(train_set)
        n_train = int(n*0.8)
        n_train_val = n
        self.train_set = self.convert(train_set[:n_train])
        self.val_set = self.convert(train_set[n_train:n_train_val])
        self.test_set = self.convert(test_set)

    def convert(self, seqs):
        if seqs is None:
            return None
        m_t = self.target + 1
        def _convert(seq):
            seq_id = seq.get('id')
            time_c = seq['time_context']
            mark_c = seq['mark_context']
            start = seq['start']
            stop = seq['stop']
            time_t = seq['time_target']
            mark_t = seq['mark_target']
            assert(util.is_sorted(time_c))
            assert(util.is_sorted(time_t))
            time = []
            mark = []
            i_c = 0
            i_t = 0
            n_c = len(time_c)
            n_t = len(time_t)
            assert(n_c == len(mark_c))
            assert(n_t == len(mark_t))
            while i_c < n_c and i_t < n_t:
                if time_t[i_t] <= time_c[i_c]:
                    time.append(time_t[i_t])
                    mark.append(mark_t[i_t])
                    i_t += 1
                else:
                    time.append(time_c[i_c])
                    mark.append(m_t + mark_c[i_c])
                    i_c += 1
            if i_t < n_t:
                time.extend(time_t[i_t:])
                mark.extend(mark_t[i_t:])
            if i_c < n_c:
                time.extend(time_c[i_c:])
                mark.extend(m_t + mark_c[i_c:])
            return {
                'id': seq_id,
                'time': time,
                'mark': mark,
                'start': start,
                'stop': stop,
                'time_test': seq.get('time_test'),
                'label_test': seq.get('label_test'),
                'lambda_x': seq.get('lambda_x'),
                't_x': seq.get('t_x'),
            }
        return [_convert(seq) for seq in seqs]

class NonContextDataLoader(ContextDataLoader):
    def convert(self, seqs):
        if seqs is None:
            return None
        def _convert(seq):
            seq_id = seq.get('id')
            start = seq['start']
            stop = seq['stop']
            time_t = seq['time_target']
            mark_t = seq['mark_target']
            assert(util.is_sorted(time_t))
            assert(len(time_t) == len(mark_t))
            return {
                'id': seq_id,
                'time': time_t,
                'mark': mark_t,
                'start': start,
                'stop': stop,
                'time_test': seq.get('time_test'),
                'label_test': seq.get('label_test'),
                'lambda_x': seq.get('lambda_x'),
                't_x': seq.get('t_x'),
            }
        return [_convert(seq) for seq in seqs]

class UnlabelContextDataLoader(ContextDataLoader):
    def convert(self, seqs):
        if seqs is None:
            return None
        m_t = self.target + 1
        def _convert(seq):
            seq_id = seq.get('id')
            time_c = seq['time_context']
            mark_c = seq['mark_context']
            start = seq['start']
            stop = seq['stop']
            time_t = seq['time_target']
            mark_t = seq['mark_target']
            assert(util.is_sorted(time_c))
            assert(util.is_sorted(time_t))
            time = []
            mark = []
            i_c = 0
            i_t = 0
            n_c = len(time_c)
            n_t = len(time_t)
            assert(n_c == len(mark_c))
            assert(n_t == len(mark_t))
            while i_c < n_c and i_t < n_t:
                if time_t[i_t] <= time_c[i_c]:
                    time.append(time_t[i_t])
                    mark.append(mark_t[i_t])
                    i_t += 1
                else:
                    time.append(time_c[i_c])
                    mark.append(m_t + mark_c[i_c])
                    i_c += 1
            if i_t < n_t:
                time.extend(time_t[i_t:])
                mark.extend(mark_t[i_t:])
            if i_c < n_c:
                time.extend(time_c[i_c:])
                mark.extend(m_t + mark_c[i_c:])
            assert(seq.get('time_test') is None)
            assert(seq.get('label_test') is None)
            if start == time_t[0]:
                # if starts with a target event
                # just use it as the start
                time_test = np.array(time_t)
            else:
                # otherwise add a start
                time_test = np.array(np.concatenate(([start], time_t)))
            # first event marks the start and is not tested
            label_test = np.zeros_like(time_test[1:])
            return {
                'id': seq_id,
                'time': time,
                'mark': mark,
                'start': start,
                'stop': stop,
                'time_test': time_test,
                'label_test': label_test,
                'lambda_x': None,
                't_x': None,
            }
        return [_convert(seq) for seq in seqs]
