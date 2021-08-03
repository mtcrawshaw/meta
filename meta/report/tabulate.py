""" Create LaTeX tables of results from training. """

from typing import Union, Dict, List

from meta.utils.metrics import Metrics


PRECISION = 3


def tabulate(
    metrics: Union[Metrics, Dict[str, Metrics]],
    table_path: str,
    tables: List[List[str]],
) -> None:
    """
    Create LaTeX tables of values from `metrics` according to `tables`, and write the
    resulting tex source code to `table_path`.
    """

    if isinstance(metrics, Metrics):
        print("Tabulation for single training runs is not yet supported.")

    # Construct LaTeX header.
    table_str = ""
    table_str += "\\documentclass{article}\n\n"
    table_str += "\\begin{document}\n\n"

    # Construct LaTeX tables.
    for metric_names in tables:

        # Table header.
        table_str += "\\begin{table}\n"
        table_str += "\\begin{center}\n"
        table_str += "\\begin{tabular}{|c||" + "c|" * len(metric_names) + "}\n"
        table_str += "\\hline\n"

        # Column names.
        for metric_name in metric_names:
            fixed_name = metric_name.replace("_", "\_")
            table_str += f" & {fixed_name}"
        table_str += " \\\\\n"
        table_str += "\\hline\\hline\n"

        # Table content.
        for method in metrics.keys():
            if method == "summary":
                continue

            table_str += method + " "
            for metric_name in metric_names:
                method_summary = metrics["summary"][metric_name][method]
                mean = method_summary["mean_performance"]
                radius = method_summary["CI_radius"]
                table_str += f"& {mean:.3f} $\\pm$ {radius:.3f}"
            table_str += " \\\\\n"

        # Table footer.
        table_str += "\\hline\n"
        table_str += "\\end{tabular}\n"
        table_str += "\\end{center}\n"
        table_str += "\\caption{Caption.}\n"
        table_str += "\\end{table}\n\n"

    # Construct LaTeX footer.
    table_str += "\\end{document}"

    # Write out LaTeX source.
    with open(table_path, "w") as table_file:
        table_file.write(table_str)
