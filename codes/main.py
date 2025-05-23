from utility.parser import args
from utility.load_data import data_generator
from trainer import STARLINE_trainer
from time import time

def run_model():
    trainer = STARLINE_trainer(data_generator=data_generator, args=args)
    best_recall, stopping_step = 0, 0
    
    ret = trainer.test()
    print(ret)
    if ret['recall'][0] > best_recall:
        best_recall = ret['recall'][0]
        trainer.save_model()

    for epoch in range(1, args['epoch'] + 1):
        t1 = time()
        train_log = trainer.train()
        
        perf_str = 'Epoch %d: time = [%.1fs], train loss = [%.5f = %.5f + %.5f + %.5f]' % (
            epoch, time() - t1, train_log['Total loss'], train_log['BPR loss'], train_log['Reg loss'], train_log['CL loss']
        )
        print(perf_str)

        if epoch % args['verbose'] == 0:
            t2 = time()
            ret = trainer.test()
            perf_str = (
                f"Validation: time = [{time() - t2:.1f}], Ks = {args['Ks']}, "
                f"recall = {[float(f'{x:.5f}') for x in ret['recall']]}, "
                f"precision = {[float(f'{x:.5f}') for x in ret['precision']]}, "
                f"hit = {[float(f'{x:.5f}') for x in ret['hit_ratio']]}, "
                f"ndcg = {[float(f'{x:.5f}') for x in ret['ndcg']]}"
            )
            print(perf_str)
            
            if ret['recall'][0] > best_recall:
                best_recall = ret['recall'][0]
                stopping_step = 0
                print('Found better model.')
                trainer.save_model()
            elif stopping_step < args['early_stopping_patience']:
                stopping_step += 1
                print('Early stopping steps: %d' % stopping_step)
            else:
                print('Early Stop!')
                break
    
    # test model
    ret = trainer.test(is_val=False)  
    print('Final result:', ret)   
            
if __name__ == '__main__':    
    if args['model_name'] == None:
        raise ValueError('model_name is None!')
    
    run_model()
