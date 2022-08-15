from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import full_check, get_list

# The source data I will test on
true_data = [0.5533834586466171, 0.21680098557568903, 0.5258004028057504, 0.37367786852322926, 0.3485319028797292,
             0.6070756829377519, 0.521097046413502, 0.2764028424405783, 0.48691056577491926, 0.2852045565356145,
             0.4799915531622847, 0.6826982541268255, 0.4230690849806399, 0.7205263157894737, 0.4010540184453228,
             0.6146167557932265, 0.42749140893470794, 0.43695652173913047, 0.6733333333333336, 0.5527426160337553,
             0.4075471698113207, 0.4267087276550999, 0.34772468714448235, 0.44823848238482383, 0.6593406593406593,
             0.5843081312410842, 0.4522031823745412, 0.3889144222814597, 0.5360651660241263, 0.5398585471110049,
             0.28083923154701723, 0.27762630312750597, 0.17466391914434304, 0.5868802106186923, 0.26065585797080193,
             0.35120842824827375, 0.29548118737001333, 0.2909532328136978, 0.21151677006270117]


class Tests2(StageTest):

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

        error = 'Incorrect new cluster centers values. Check your calculate_new_centers function.'
        check_result = full_check(student, true_data, '', tolerance=0.1, error_str=error)
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests2().run_tests()
