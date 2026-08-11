#!/usr/bin/env python3

import logging
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Creates and manages a timestamped logging directory and logger instance.

    For each execution, a new directory is created under the specified
    base directory. All log messages are written to a file named
    ``run.log`` inside this directory.

    Attributes
    ----------
    run_dir : pathlib.Path
        Path to the run-specific output directory.

    logger : logging.Logger
        Configured logger instance.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is ``"validation2d"``.

    base_dir : str, optional
        Base directory in which run directories are created.
        Default is ``"runs"``.

    run_prefix : str, optional
        Prefix used for naming run directories.
        Default is ``"validation2d"``.

    level : int, optional
        Logging level from the ``logging`` module.
        Default is ``logging.DEBUG``.

    Examples
    --------
    >>> log_manager = Logger()
    >>> logger = log_manager.get_logger()
    >>> logger.info("Starting validation")

    This creates a directory similar to:

    runs/
    └── validation2d_20260720_104530/
        └── run.log

    You can also customize the logger:

    >>> log_manager = Logger(
    ...     name="training",
    ...     run_prefix="experiment"
    ... )
    >>> logger = log_manager.get_logger()
    >>> logger.info("Training started")
    """

    def __init__(
        self,
        name: str = "validation2d",
        base_dir: str = "runs",
        run_prefix: str = "validation2d",
        level: int = logging.DEBUG,
    ):
        """
        Initialize a logger and create a timestamped run directory.

        Parameters
        ----------
        name : str
            Name of the logger.

        base_dir : str
            Parent directory where run folders are created.

        run_prefix : str
            Prefix for the generated run folder.

        level : int
            Logger verbosity level.

        Examples
        --------
        >>> log_manager = Logger()
        >>> log_manager.run_dir
        PosixPath('runs/validation2d_20260720_104530')

        >>> custom = Logger(
        ...     name="my_app",
        ...     base_dir="outputs",
        ...     run_prefix="test"
        ... )
        """
        # Create timestamped run directory
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{run_prefix}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            log_file = self.run_dir / "run.log"

            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
            )

            self.logger.addHandler(fh)
            self.logger.propagate = False

    def get_logger(self) -> logging.Logger:
        """
        Return the configured logger instance.

        Returns
        -------
        logging.Logger
            Logger object that can be used to write messages.

        Examples
        --------
        >>> log_manager = Logger()
        >>> logger = log_manager.get_logger()

        >>> logger.debug("Debug message")
        >>> logger.info("Information message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
        """
        return self.logger

    def get_run_dir(self) -> Path:
        """
        Return the run-specific output directory.

        Returns
        -------
        pathlib.Path
            Path to the directory where logs and other run artifacts
            can be stored.

        Examples
        --------
        >>> log_manager = Logger()
        >>> run_dir = log_manager.get_run_dir()
        >>> print(run_dir)

        Store additional outputs:

        >>> results_file = run_dir / "results.csv"
        >>> model_file = run_dir / "model.pt"
        """
        return self.run_dir