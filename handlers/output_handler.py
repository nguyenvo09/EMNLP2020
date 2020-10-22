import sys

class FileHandler(object):
    mylogfile = None
    mylogfile_details = None
    error_analysis_log_validation = None
    error_analysis_log_testing = None
    error_analysis_log_test2 = None
    error_analysis_log_test3 = None


    def __init__(self):
        pass

    @classmethod
    def init_log_files(cls, log_file):
        if log_file != None:
            cls.mylogfile = open(log_file, "w")
            cls.mylogfile_details = open(log_file + "_best_details.json", "w")
            cls.error_analysis_log_validation = open(log_file + "_error_analysis_validation.json", "w")
            cls.error_analysis_log_testing = open(log_file + "_error_analysis_testing.json", "w")
            cls.error_analysis_log_test2 = open(log_file + "_error_analysis_test2.json", "w")
            cls.error_analysis_log_test3 = open(log_file + "_error_analysis_test3.json", "w")

    @classmethod
    def myprint(cls, message):
        assert cls.mylogfile != None, "The LogFile is not initialized yet!"
        print(message)
        sys.stdout.flush()
        if cls.mylogfile != None:
            print(message, file = cls.mylogfile)
            cls.mylogfile.flush()

    @classmethod
    def myprint_details(cls, message):
        assert cls.mylogfile_details != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if cls.mylogfile_details != None:
            print(message, file = cls.mylogfile_details)
            cls.mylogfile_details.flush()

    @classmethod
    def save_error_analysis_validation(cls, message: str):
        assert cls.error_analysis_log_validation != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if cls.error_analysis_log_validation != None:
            print(message, file = cls.error_analysis_log_validation)
            cls.error_analysis_log_validation.flush()

    @classmethod
    def save_error_analysis_testing(cls, message: str):
        assert cls.error_analysis_log_testing != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if cls.error_analysis_log_testing != None:
            print(message, file = cls.error_analysis_log_testing)
            cls.error_analysis_log_testing.flush()

    @classmethod
    def save_error_analysis_test2(cls, message: str):
        assert cls.error_analysis_log_test2 != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if cls.error_analysis_log_test2 != None:
            print(message, file=cls.error_analysis_log_test2)
            cls.error_analysis_log_test2.flush()

    @classmethod
    def save_error_analysis_test3(cls, message: str):
        assert cls.error_analysis_log_test3 != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if cls.error_analysis_log_test3 != None:
            print(message, file=cls.error_analysis_log_test3)
            cls.error_analysis_log_test3.flush()