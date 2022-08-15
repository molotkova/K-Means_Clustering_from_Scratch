from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List

# The source data I will test on
true_data = 3


class Tests5(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed.")

        # Getting the student's results from the reply
        if not reply.isnumeric():
            return CheckResult.wrong('Seems that data output is in wrong format.')
        
        student = int(reply)

        if student != true_data:
            return CheckResult.wrong('Incorrect k value. Check your k-find function.')

        return CheckResult.correct()


if __name__ == '__main__':
    Tests5().run_tests()
