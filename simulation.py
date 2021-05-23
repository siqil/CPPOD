import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import util
import copy

def merge(a, b, idx=None):
    i = 0
    j = 0
    n = len(a)
    m = len(b)
    c = []
    while i < n and j < m:
        if a[i] <= b[j]:
            c.append(a[i])
            if idx is not None:
                idx.append(0)
            i += 1
        else:
            c.append(b[j])
            if idx is not None:
                idx.append(1)
            j += 1
    if i < n:
        c.extend(a[i:])
        if idx is not None:
            idx.extend([0]*len(a[i:]))
    elif j < m:
        c.extend(b[j:])
        if idx is not None:
            idx.extend([1]*len(b[j:]))
    return c

class MJPSim:
    def __init__(self, q, param):
        self.q = q
        self.param = param

    def sim(self, t_max, dt):
        vt_z, vz = self.sim_mjp(t_max=t_max)
        vt_event, lambda_x, t_x = self.sim_target(self.param, vt_z, vz, t_max=t_max, dt=dt)
        return {
            'start': 0,
            'stop': t_max,
            'time_context': vt_z,
            'mark_context': vz,
            'time_target': vt_event,
            'mark_target': np.ones_like(vt_event, dtype=np.int32),
            'lambda_x': lambda_x,
            't_x': t_x,
        }

    def sim_mjp(self, t_max):
        q = self.q
        m = q.shape[0]
        assert(np.all(np.sum(q, axis=1) == 0))
        states = np.arange(0, m)
        stay = np.diag(q)
        trans = q.copy()
        np.fill_diagonal(trans, 0)
        trans = trans / np.sum(trans, axis=1)
        vt_z = [0]
        vz = [0]
        s = 0
        t = 0
        while True:
            t += np.random.exponential(-1/stay[s])
            if t <= t_max:
                s = np.random.choice(states, p=trans[s,:])
                vt_z.append(t)
                vz.append(s)
            else:
                break
        return np.array(vt_z), np.array(vz, dtype=np.int32)

    def sim_target(self, param, vt_z, vz, t_max, dt):
        raise NotImplementedError

    def sim_next(self, lambda_t, lambda_max, t_beg, t_end):
        t_next = t_beg
        while True:
            t_next += np.random.exponential(1 / lambda_max)
            if (t_next > t_end) or (np.random.uniform() * lambda_max <= lambda_t(t_next)):
                return t_next

class PoisMJPSim(MJPSim):
    def sim_target(self, lambda_, vt_z, vz, t_max, dt):
        t_x = np.arange(dt, t_max, dt)
        lambda_x = np.zeros_like(t_x)
        t = 0
        vt_event=[]
        vt_z = np.append(vt_z, t_max)
        for k in range(len(vt_z)-1):
            t_l = vt_z[k + 1]
            lambda_x[(t_x > t) & (t_x <= t_l)] = lambda_[vz[k]]
            lambda_t = lambda t: lambda_[vz[k]]
            lambda_max = lambda_[vz[k]]
            while True:
                t = self.sim_next(lambda_t, lambda_max, t, t_l)
                if t <= t_l:
                    vt_event.append(t)
                else:
                    break
            t = t_l
        vt_event = np.array(vt_event)
        return vt_event, lambda_x, t_x

class PoisSinMJPSim(MJPSim):
    def sim_target(self, param, vt_z, vz, t_max, dt):
        t_x = np.arange(dt, t_max, dt)
        lambda_x = np.zeros_like(t_x)
        t = 0
        vt_event=[]
        vt_z = np.append(vt_z, t_max)
        for k in range(len(vt_z)-1):
            t_l = vt_z[k + 1]
            idx = (t_x > t) & (t_x <= t_l)
            lambda_t = lambda t: param[vz[k]] * (1+np.sin(t))
            lambda_x[idx] = lambda_t(t_x[idx])
            lambda_max = param[vz[k]]
            while True:
                t = self.sim_next(lambda_t, lambda_max, t, t_l)
                if t <= t_l:
                    vt_event.append(t)
                else:
                    break
            t = t_l
        vt_event = np.array(vt_event)
        return vt_event, lambda_x, t_x

class GamMJPSim(MJPSim):
    def sim_target(self, param, vt_z, vz, t_max, dt):
        t_x = np.arange(dt, t_max, dt)
        lambda_x = np.zeros_like(t_x)
        t = 0
        t_prev = 0
        vt_event = []
        vt_z = np.append(vt_z, t_max)
        lambda_event = []
        for k in range(len(vt_z)-1):
            a = param[vz[k],0]
            b = 1/param[vz[k],1]
            step = a*b
            if vt_z[k+1] - t - step < 10*dt:
                t_l = vt_z[k+1]
            else:
                t_l = t+step
            while True:
                idx = (t_x > t) & (t_x <= t_l)
                def lambda_t(t):
                    return stats.gamma.pdf(t - t_prev, a, scale=b) / stats.gamma.sf(t - t_prev, a, scale=b)
                lambda_x[idx] = lambda_t(t_x[idx])
                if a >= 1:
                    lambda_max = lambda_t(t_l)
                else:
                    lambda_max = lambda_t(t)
                assert(lambda_max < np.inf)
                if lambda_max == 0: # avoid overflow in exponential
                    t = t_l + 1
                else:
                    t = self.sim_next(lambda_t, lambda_max, t, t_l)
                if t <= t_l:
                    vt_event.append(t)
                    lambda_event.append(lambda_t(t))
                    t_prev = t
                elif t_l >= vt_z[k + 1]:
                    break
                else:
                    t = t_l
                    if vt_z[k+1] - t - step < 10*dt:
                        t_l = vt_z[k+1]
                    else:
                        t_l = t+step
            t = t_l
        vt_event = np.array(vt_event)
        lambda_event = np.array(lambda_event)
        t_x = np.concatenate((t_x, vt_event))
        lambda_x = np.concatenate((lambda_x, lambda_event))
        idx = np.argsort(t_x)
        t_x = t_x[idx]
        lambda_x = lambda_x[idx]
        return vt_event, lambda_x, t_x

class OmissSim:
    def __init__(self, w, rate_omiss=0.1, regulator=None):
        # regulator is a function which changes the rate over time
        self.rate_omiss = rate_omiss
        self.w = w
        self.regulator = regulator

    def sim(self, seq):
        vt_event = seq['time_target']
        t_max = seq['stop']
        t_min = seq['start']
        vt_event, vt_omiss = self.sim_omiss(vt_event, t_min)
        vt_test = self.gen_test(vt_event, t_min, t_max)
        vlabel = self.gen_label(vt_test, vt_event, vt_omiss)
        seq = seq.copy()
        seq.update({
            'time_target': vt_event,
            'mark_target': np.ones_like(vt_event, dtype=np.int32),
            'time_test': vt_test,
            'label_test': vlabel,
            'time_omiss': vt_omiss,
            'mark_omiss': np.ones_like(vt_omiss, dtype=np.int32),
        })
        return seq

    def sim_omiss(self, vt_event, t_min):
        n = len(vt_event)
        if self.regulator is None:
            rate = self.rate_omiss
        else:
            rate = self.rate_omiss * self.regulator(vt_event)
        trials = np.random.binomial(1, rate, n)
        # always keep the event at t_min
        if vt_event[0] == t_min:
            trials[0] = 0
        idx_omiss = np.nonzero(trials)
        vt_omiss = vt_event[idx_omiss]
        vt_event_left = np.delete(vt_event, idx_omiss)
        return vt_event_left, vt_omiss

    def gen_test(self, vt_event, t_min, t_max):
        w = self.w
        vt_test = []
        vt = vt_event
        # we ignore events at t_min but keep events at t_max
        if len(vt) > 0 and vt[0] == t_min:
            vt = np.concatenate((vt, [t_max]))
        else:
            vt = np.concatenate(([t_min], vt, [t_max]))
        n = len(vt)
        for i in range(n-1):
            t = vt[i]
            vt_test.append(vt[i])
            while vt[i+1] > t + w:
                t_next = t + np.random.uniform(0, w)
                vt_test.append(t_next)
                t = t_next
        vt_test = np.array(vt_test)
        return vt_test

    def gen_label(self, vt, vt_event, vt_omiss):
        n = len(vt)
        vlabel = np.zeros(n-1)
        for i in range(n-1):
            t_beg = vt[i]
            t_end = vt[i+1]
            if i == 0:
                vlabel[i] = np.any((vt_omiss >= t_beg) & (vt_omiss <= t_end))
            else:
                vlabel[i] = np.any((vt_omiss > t_beg) & (vt_omiss <= t_end))
        return vlabel


class CommissSim:
    def __init__(self, rate=0.1, shrink=1, regulator=None):
        self.rate = rate
        self.shrink = shrink
        self.regulator = regulator

    def sim(self, seq):
        vt_event = seq['time_target']
        t_max = seq['stop']
        t_min = seq['start']
        vt_event, vlabel = self.sim_commiss(vt_event, t_min, t_max)
        # skip the event at t_min
        vt_test = vt_event
        if vt_test[0] == t_min and vlabel[0] == 0:
            vt_test = vt_test[1:]
            vlabel = vlabel[1:]
        # padding
        vt_test = np.concatenate(([t_min], vt_test))
        seq = seq.copy()
        seq.update({
            'time_target': vt_event,
            'mark_target': np.ones_like(vt_event, dtype=np.int32),
            'time_test': vt_test,
            'label_test': vlabel,
        })
        return seq

    def sim_commiss(self, vt_event, t_min, t_max):
        rate = self.rate
        shrink = self.shrink
        if shrink < 1:
            inter_event = np.diff(np.concatenate((vt_event, [t_max])))
            inter_event *= shrink
            total_inter_event = inter_event.sum()
            m = np.random.poisson(total_inter_event * rate, 1)
            vt_commiss = np.random.uniform(0, total_inter_event, m)
            cum_inter_event = np.cumsum(inter_event)
            for i in range(m):
                j = np.argwhere(cum_inter_event > vt_commiss[i])[0]
                if j > 0:
                    tmp = vt_commiss[i] - cum_inter_event[j-1]
                else:
                    tmp = vt_commiss[i]
                tmp += vt_event[j]
                assert(tmp >= vt_event[0])
                vt_commiss[i] = tmp
        else:
            m = np.random.poisson((t_max - t_min) * rate, 1)
            vt_commiss = np.random.uniform(t_min, t_max, m)
            if self.regulator is not None:
                p = self.regulator(vt_commiss)
                keep = (np.random.binomial(1, p) > 0)
                vt_commiss = vt_commiss[keep]
        vt_commiss = np.sort(vt_commiss)
        vlabel = []
        vt_event = np.array(merge(vt_event, vt_commiss, vlabel))
        vlabel = np.array(vlabel)
        assert(util.is_sorted(vt_event))
        return vt_event, vlabel

def compute_empirical_rate(seqs):
    t = 0
    n = 0
    for seq in seqs:
        t += seq['stop'] - seq['start']
        n += len(seq['time_target'])
    return n/t

def sim_data_test_omiss(data_train, data_test, p=0.1, seed=0, regulator=None, regulator_generator=None):
    # generate test_omiss
    np.random.seed(seed)
    data_test_omiss = copy.deepcopy(data_test)
    n_test = len(data_test)
    w = 2 / compute_empirical_rate(data_train)
    if regulator_generator is None:
        omiss_sim = OmissSim(w, p, regulator=regulator)
        for i in range(n_test):
            data_test_omiss[i] = omiss_sim.sim(data_test_omiss[i])
    else:
        for i in range(n_test):
            regulator = regulator_generator()
            omiss_sim = OmissSim(w, p, regulator=regulator)
            data_test_omiss[i] = omiss_sim.sim(data_test_omiss[i])
    return data_test_omiss

def sim_data_test_commiss(data_train, data_test, alpha=0.1, seed=0, regulator=None, regulator_generator=None):
    # generate test_commiss
    np.random.seed(seed)
    data_test_commiss = copy.deepcopy(data_test)
    n_test = len(data_test)
    rate = compute_empirical_rate(data_test)
    if regulator_generator is None:
        commiss_sim = CommissSim(alpha * rate, 1, regulator=regulator)
        for i in range(n_test):
            data_test_commiss[i] = commiss_sim.sim(data_test_commiss[i])
    else:
        for i in range(n_test):
            regulator = regulator_generator()
            commiss_sim = CommissSim(alpha * rate, 1, regulator=regulator)
            data_test_commiss[i] = commiss_sim.sim(data_test_commiss[i])
    return data_test_commiss

def create_rand_pc_regulator(step, t_min, t_max):
    m = np.floor((t_max - t_min) / step).astype(int)
    p = np.random.uniform(size=m)
    def regulator(t):
        i = np.floor((t - t_min) / step).astype(int)
        return p[i]
    return regulator

def sparse_rand_pc_regulator(t, step):
    i = np.floor(t / step).astype(int)
    u = np.unique(i)
    p = np.random.uniform(size=len(u))
    r = np.zeros_like(t)
    for k, v in enumerate(u):
        r[i == v] = p[k]
    return r

def plot_events(seq):
    vt_event = seq['time_target']
    lambda_x = seq['lambda_x']
    t_x = seq['t_x']
    vt_omiss = seq.get('time_omiss')
    scale = 0.25 * np.max(lambda_x)
    plt.figure()
    if vt_omiss is None:
        vlabel = seq.get('label_test')
        if vlabel is None:
            plt.plot(t_x,lambda_x)
            plt.stem(vt_event, scale*np.ones_like(vt_event), 'k-', 'ko')
        else:
            plt.plot(t_x,lambda_x)
            plt.stem(vt_event[vlabel==0], scale*np.ones_like(vt_event[vlabel==0]), 'k-', 'ko')
            if any(vlabel):
                plt.stem(vt_event[vlabel==1], scale*np.ones_like(vt_event[vlabel==1]), 'r-', 'ro')
    else:
        plt.plot(t_x,lambda_x)
        if len(vt_event) > 0:
            plt.stem(vt_event, scale*np.ones_like(vt_event), 'k-', 'ko')
        if len(vt_omiss) > 0:
            plt.stem(vt_omiss, scale*np.ones_like(vt_omiss), 'r-', 'ro')


if __name__=='__main__':
    # t_max = 100
    # dt = 0.01
    # q = np.array([
    #     [-0.1, 0.05, 0.05],
    #     [0.05, -0.1, 0.05],
    #     [0.05, 0.05, -0.1]
    # ])
    # param = np.array([.1, .1, .2])
    # w = 2 / np.max(param)
    # pois_sim = PoisMJPSim(q, param)
    # seq = pois_sim.sim(t_max, dt)
    # plot_events(seq)
    # omiss_sim = OmissSim(w, 0.1)
    # seq_omiss = omiss_sim.sim(seq)
    # plot_events(seq_omiss)
    # commiss_sim = CommissSim(shrink=1)
    # seq_commiss = commiss_sim.sim(seq)
    # plot_events(seq_commiss)
    # plt.show()

    t_max = 100
    dt = 0.01
    q = np.array([
        [-0.1, 0.05, 0.05],
        [0.05, -0.1, 0.05],
        [0.05, 0.05, -0.1]
    ])
    param = np.array([.1, .2, .3])
    w = 2 / np.max(param)
    # regulator = lambda t: (1 + np.sin(t/100*2*np.pi))/2
    regulator = create_rand_pc_regulator(20, 0, t_max)
    # regulator = lambda t: sparse_rand_pc_regulator(t, 20)
    x = np.arange(0, t_max, dt)
    y_reg = regulator(x)
    pois_sim = PoisSinMJPSim(q, param)
    seq = pois_sim.sim(t_max, dt)
    plot_events(seq)
    omiss_sim = OmissSim(w, 0.9)
    seq_omiss = omiss_sim.sim(seq)
    plot_events(seq_omiss)
    omiss_sim = OmissSim(w, 1, regulator=regulator)
    seq_omiss = omiss_sim.sim(seq)
    plot_events(seq_omiss)
    plt.plot(x, y_reg, 'g--')
    rate = compute_empirical_rate(([seq]))
    commiss_sim = CommissSim(rate=0.9*rate)
    seq_commiss = commiss_sim.sim(seq)
    plot_events(seq_commiss)
    commiss_sim = CommissSim(rate=1*rate, regulator=regulator)
    seq_commiss = commiss_sim.sim(seq)
    plot_events(seq_commiss)
    plt.plot(x, y_reg, 'g--')
    plt.show()

    # t_max = 100
    # dt = 0.01
    # q = np.array([
    # [-0.05, 0.05],
    # [0.05, -0.05]
    # ])
    # param = np.array([
    #     [100., 10.],
    #     [50., 10.]])
    # w = 2 * np.min(param[:,0]/param[:,1])
    # gam_sim = GamMJPSim(q, param)
    # seq = gam_sim.sim(t_max, dt)
    # plot_events(seq)
    # plt.vlines(seq['time_context'], 0, np.max(seq['lambda_x']), linestyles='dashed')
    # omiss_sim = OmissSim(w, 0.1)
    # seq_omiss = omiss_sim.sim(seq)
    # plot_events(seq_omiss)
    # commiss_sim = CommissSim(shrink=1)
    # seq_commiss = commiss_sim.sim(seq)
    # plot_events(seq_commiss)
    # plt.show()
