import faulthandler

from Experiment import Experiment

if __name__ == '__main__':
    faulthandler.enable()
    exp1 = Experiment()
    # Executar o experimento principal
    exp1.plan_experiment()
