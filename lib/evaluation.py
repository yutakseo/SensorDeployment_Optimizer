import os
__file__ = os.getcwd()
__root__ = os.path.dirname(__file__)


class CornerEvaluation():
    def accuracy(self, result, correct):
        '''
        Arguments
        ---------
        result : corner detection result
            | array
        correct : correct set of corners
            | array
        '''

        '''
        >>>>>>>>>>
        여기에 Corner Detection의 정확도를 평가하는 코드를 넣으면 되겠다.
        <<<<<<<<<<
        '''

        return accuracy_score