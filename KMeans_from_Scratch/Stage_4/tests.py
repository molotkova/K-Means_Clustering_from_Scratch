from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import full_check, get_list

# The source data I will test on
true_data = [9.777501612616108, 5.631533557769523, 4.018301201966832, 3.3388388533650306, 2.8921917593894824,
             2.5141083632201937, 2.2585654059801263, 2.089139663340243, 1.9430146711949676, 1.8243300004053062]


class Tests4(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed!")

        if reply.count('[') != 1 or reply.count(']') != 1:
            return CheckResult.wrong('No expected list was found in output!')

        # Getting the student's results from the reply

        try:
            student, _ = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that data output is in wrong format!')

        error = 'Incorrect error values. Check how you calculate the error for each k.'
        check_result = full_check(student, true_data, '', tolerance=0.1, error_str=error)
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests4().run_tests()
