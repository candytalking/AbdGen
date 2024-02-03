"""
This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
Code for
Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
https://arxiv.org/abs/2310.17451

"""

import os
import asyncio
import re
import time


class AbdTrainer:

    def __init__(self, config, pl_tmp_path):
        self.swipl = config['swipl']
        self.dataset = config['dataset']
        self.sem = asyncio.Semaphore(config['num_cpu_core'])
        self.loop = asyncio.get_event_loop()
        self.bk_file = config['bk_file']
        self.label_names = config['label_names']
        self.label_mapping = config['grounding_to_label_table_str']
        self.pl_tmp_path = pl_tmp_path
        self.abd_time_limit = config['abd_time_limit']

    def gen_abd_variables_mario(self, pos_names, neg_names):
        # variable names[X1,X2] and examples string [f([X1,X2,..],y),...]
        pos_sample_str, neg_sample_str = "[", "["

        for name_list in pos_names:
            pos_sample_str += "f(["
            for name in name_list:
                pos_sample_str += ("'" + name + "',")
            pos_sample_str = pos_sample_str[:-1] + "],[]),"
            haha = 1
        pos_sample_str = pos_sample_str[:-1] + "],"

        for name_list in neg_names:
            neg_sample_str += "f(["
            for name in name_list:
                neg_sample_str += ("'" + name + "',")
            neg_sample_str = neg_sample_str[:-1] + "],[]),"
            haha = 1
        neg_sample_str = neg_sample_str[:-1] + "],"
        return pos_sample_str, neg_sample_str


    # Generate probabilistic facts
    def gen_prob_facts_mario(self, grounding):
        nn_probs, term_grds, var_names = grounding[0][0], grounding[1], grounding[2][0]
        prob_facts_str = ""
        for i in range(len(nn_probs)):
            case_probs, case_names = nn_probs[i], var_names[i]
            for j in range(case_probs.shape[0]):
                vname = case_names[j]
                if j < case_probs.shape[0]-1:
                    prob = case_probs[j]
                    for k in range(len(self.label_names)):
                        lname = self.label_names[k]
                        if prob[k] > 0:
                            fact_str = "nn('{}',{},{}).\n".format(vname, lname, prob[k])
                            prob_facts_str = prob_facts_str + fact_str
                else:
                    lname = list(term_grds[i])
                    fact_str = "nn('{}',{},{}).\n".format(vname, lname, 1.0)
                    prob_facts_str = prob_facts_str + fact_str
                prob_facts_str = prob_facts_str + "\n"
        return prob_facts_str

    # Run prolog to get the output.
    # Return the STDOUT and error codes (-1 for runtime error, -2 for timeout)
    async def run_pl(self, file_path):
        cmd = self.swipl+ " --stack-limit=8g -s {} -g a -t halt".format(file_path)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        try:
            # 2 seconds timeout
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.abd_time_limit)
            if proc.returncode == 0:
                return 0, stdout.decode('UTF-8')
            else:
                return -1, stderr.decode('UTF-8')  # runtime error
        except asyncio.TimeoutError as e:
            if proc.returncode is None:
                try:
                    proc.kill()
                except OSError:
                    # Ignore 'no such process' error
                    pass
#                proc.kill()
            return -2, "Timeout " + str(e)  # timeout error

    def parse_pl_result_mario(self, pl_out_str):
        prog_str, labels_str = self.read_pl_out(pl_out_str)
        if self.dataset == 'mario':
            sep_char, var_length, grd_length = 'P', 5, 5
        else:
            raise NameError('unrecognized dataset')
        vars = [[labels_str[i:i+var_length], labels_str[i+var_length+1:i+var_length+grd_length+1]]
                for i in [m.start() for m in re.finditer(sep_char, labels_str)]]
        abd_labels = [self.label_mapping[vars[len(vars)-i-1][-1]] for i in range(len(vars))]
        return vars, prog_str, abd_labels

    def read_pl_out(self, pl_out_str):
        prog = ""
        labels = None

        prog_start = False
        label_start = False
        for line in pl_out_str.splitlines():
            if line[0] == '-':
                if line[2:-1] == 'Program':
                    prog_start = True
                    continue
                elif line[2:-1] == 'Abduced Facts':
                    prog_start = False
                    label_start = True
                    continue
            if prog_start:
                prog = prog + line + "\n"
            if label_start:
                labels = line
        if labels is not None:
            labels = labels[1:-1]
        else:
            haha = 1
        return prog, labels

    # Perceives the sample of images in each example and convert the result to a Prolog background knowledge file
    def perception_to_kb(self, grounding, pl_file_path):
        header = ":- ['{}'].\n\n".format(self.bk_file)
    #    # use perception model to calculate the probabilistic distribution p(z|x)
        # 1a. generate variable list
        # 1b. generate string of metagol example "[f(['X1','X2'],y1),f(['X3','X4'],y2)...]"
        if self.dataset == 'mario':
            pos_sample_str, neg_sample_str = self.gen_abd_variables_mario(grounding[2][0], grounding[3][0])
            # 2. generate string of all "nn('X1',1,p1).\nnn('X1',2,p2)..."
            prob_facts = self.gen_prob_facts_mario(grounding)
        else:
            raise NameError('unrecognized dataset')

        # generate string of query "learn :- Pos=..., metaabd(Pos,[])."
        query_str = "\na :- Pos={} Neg={} metaabd(Pos,Neg).\n".format(pos_sample_str, neg_sample_str)
        with open(pl_file_path, 'w') as pl:
            pl.write(header)
            pl.write(prob_facts)
            pl.write(query_str)

    async def abduce_coroutine_concurrent(self, i, grounding):
    #    start = time.time()
        pl_file_path = os.path.join(self.pl_tmp_path, '{}_bk.pl'.format(i))
        self.perception_to_kb(grounding, pl_file_path=pl_file_path)
        _, pl_out = await self.run_pl(file_path=pl_file_path)
    #    pl_err, pl_out = await run_pl(file_path=pl_file_path, timeout=timeout)
        return pl_out

    async def safe_abduce_concurrent(self, i, grounding):
        async with self.sem:  # semaphore limits num of simultaneous downloads
            return await self.abduce_coroutine_concurrent(i, grounding)

    async def abduce_concurrent(self, groundings_for_abduce):
        tasks = [
            # creating task starts coroutine
            asyncio.ensure_future(self.safe_abduce_concurrent(i, grounding))
            for i, grounding
            in enumerate(groundings_for_abduce)
        ]
        return await asyncio.gather(*tasks)  # await moment all tasks done

    def abduce_coroutine(self, i, grounding):
        pl_file_path = os.path.join(self.pl_tmp_path, '{}_bk.pl'.format(i))
        self.perception_to_kb(grounding, pl_file_path=pl_file_path)
        pl_out = self.run_pl(file_path=pl_file_path)
        return pl_out

    def safe_abduce(self, i, grounding):
        return self.abduce_coroutine(i, grounding)

    def abduce(self, groundings_for_abduce):
        result, task_time = [], []
        for i, grounding in enumerate(groundings_for_abduce):
            start = time.time()
            result.append(self.safe_abduce(i, grounding))
            task_time.append(time.time() - start)
        return result, task_time

    def train_abduce(self, groundings_for_abduce, parallel=True, epochs=10, log_interval=500, **kwargs):
        # Start concurrent abduction
        start = time.time()
        if parallel:
            try:
                tasks_results = self.loop.run_until_complete(self.abduce_concurrent(groundings_for_abduce))
            finally:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        else:
            tasks_results, task_time = self.abduce(groundings_for_abduce)
        end = time.time()
        gap = end - start
        progs, abd_grds, abd_labels = [], [], []
        for result in tasks_results:
            # gather the outputs
            if result[:5] != 'ERROR' and result[:7] != 'Timeout':
                if self.dataset == 'mario':
                    abd_grd, prog_str, abd_label = self.parse_pl_result_mario(result)
                    abd_grds.append(abd_grd)
                    abd_labels.append(abd_label)
                    progs.append(prog_str)
            else:
                abd_grds.append(None)
                abd_labels.append(None)
                progs.append(None)
        return progs, abd_grds, abd_labels
