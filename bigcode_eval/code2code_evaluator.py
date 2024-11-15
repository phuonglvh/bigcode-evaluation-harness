from .evaluator import Evaluator

class Code2CodeEvaluator(Evaluator):
    def __init__(self, accelerator, model, tokenizer, args):
        uper().__init__(accelerator, model, tokenizer, args)
        
    def generate_text(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) -
                      self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(
            self.args.limit_start, self.args.limit_start+n_tasks)]
