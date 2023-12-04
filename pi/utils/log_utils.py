# Created by shaji at 03/11/2023


import datetime


def create_file(exp_output_path, file_name):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    file_name = str(exp_output_path / f"log_{date_now}_{time_now}_{file_name}.txt")
    with open(file_name, "w") as f:
        f.write(f"{file_name} from {date_now}, {time_now}\n")

    return str(exp_output_path / file_name)


def create_log_file(log_path, exp_name):


    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    file_name = str(log_path / f"log_{exp_name}_{date_now}_{time_now}.txt")
    with open(file_name, "w") as f:
        f.write(f"Log from {date_now}, {time_now}")

    log_file = str(log_path / file_name)
    add_lines(f"- log_file_path:{log_file}", log_file)
    return log_file


def add_lines(line_str, log_file):
    print(line_str)
    with open(log_file, "a") as f:
        f.write(str(line_str) + "\n")


def write_clause_to_file(clauses, pi_clause_file):
    with open(pi_clause_file, "a") as f:
        for c in clauses:
            f.write(str(c) + "\n")


def write_predicate_to_file(invented_preds, inv_predicate_file):
    with open(inv_predicate_file, "a") as f:
        for inv_pred in invented_preds:
            arg_str = "("
            for a_i, a in enumerate(inv_pred.args):
                arg_str += str(a)
                if a_i != len(inv_pred.args) - 1:
                    arg_str += ","
            arg_str += ")"
            head = inv_pred.name + arg_str
            for body in inv_pred.body:
                clause_str = head + ":-" + str(body).replace(" ", "")[1:-1] + "."
                print(str(clause_str))
                f.write(str(clause_str) + "\n")


