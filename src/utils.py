import os
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

def get_base_filename(file_path):
    """
    Extract base filename (without extension) from file path
    
    Args:
        file_path: File path
        
    Returns:
        Base filename
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def create_directories(*dirs):
    """
    Create multiple directories
    
    Args:
        *dirs: List of directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def print_header(text, char='=', color=Fore.CYAN):
    """
    Print a formatted header
    
    Args:
        text: Header text
        char: Character to use for the line
        color: Color for the header (if None, default color is used)
    """
    line = char * 50
    color_code = color if color is not None else ""
    print(f"\n{color_code}{line}")
    print(f"{color_code}{text}")
    print(f"{color_code}{line}")

def print_success(text):
    """Print success message in green"""
    print(f"{Fore.GREEN}{text}")

def print_info(text):
    """Print info message in blue"""
    print(f"{Fore.BLUE}{text}")

def print_warning(text):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}{text}")

def print_error(text):
    """Print error message in red"""
    print(f"{Fore.RED}{text}")

def print_result(label, value, color=Fore.MAGENTA):
    """Print a labeled result with custom color"""
    print(f"{color}{label}: {Style.RESET_ALL}{value}")

def print_table(df, title=None):
    """
    Print a formatted table from DataFrame
    
    Args:
        df: DataFrame to print
        title: Optional title for the table
    """
    if title:
        print(f"\n{Fore.CYAN}{title}:")
    
    col_widths = [max(len(str(x)) for x in df[col].astype(str)) for col in df.columns]
    col_widths = [max(len(col), width) for col, width in zip(df.columns, col_widths)]
    
    header = " | ".join(f"{col:{width}s}" for col, width in zip(df.columns, col_widths))
    print(f"{Fore.GREEN}{header}")
    
    separator = "-+-".join("-" * width for width in col_widths)
    print(f"{Fore.GREEN}{separator}")
    
    for _, row in df.iterrows():
        row_str = " | ".join(f"{str(val):{width}s}" for val, width in zip(row.values, col_widths))
        print(f"{row_str}")
