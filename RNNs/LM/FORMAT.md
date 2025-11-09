I. Commenting Style
Comments exist to clarify the function of the code, not to hold a conversation. They should be professional, impersonal, and efficient.
Guiding Principle
Be functional, not conversational. Comments should explain the "what" and the functional "why" (e.g., the purpose of an algorithm choice), but not the author's thought process or journey.
What to Do ✅
Comment on non-obvious code: Explain complex lines, dense calculations, or the purpose of specific parameter choices.
Python
# Set x-limit to the 99.5th percentile to exclude extreme outliers from view.
facet_kws=dict(xlim=(0, np.percentile(dist_df['distance_nm'], 99.5)))


Explain the purpose of code blocks: Use comments to introduce sections of code that perform a distinct step in the logic.
Python
# Use the tail of the distribution to estimate the Gamma (noise) parameters.
tail_region = distances[distances > 2 * mode_guess]


Use section headers for script structure: Use clear, capitalized headers to break the script into logical, navigable parts.
Python
# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


What to Avoid ❌
Do not use first or second person: The code should not address the reader or refer to the author.
Instead of: # I'm using this to find neighbors because it's fast.
Use: # Build k-d trees for efficient nearest-neighbor searches.
Do not write conversational filler: Avoid phrases that don't describe the code's function.
Instead of: # Now, let's get the results and plot them.
Use: # --- GENERATE PLOTS ---
Do not state the obvious: A comment should add information that the code itself cannot convey.
Instead of: # Loop through each movie.
On this code: for movie in MOVIE_NUMBERS: (This line needs no comment).

II. Naming Conventions
Variable and function names are a primary form of documentation. They should be clear and unambiguous.
Be descriptive: Names should accurately describe the data or action they represent. Clarity is more important than brevity.
Good: mean_distance_nm, load_picasso_data
Bad: md, get_data
Use standard casing:
snake_case for variables and functions (e.g., search_range).
UPPERCASE_SNAKE_CASE for global constants (e.g., BASE_FOLDER).
PascalCase for classes (e.g., class DataAnalyzer:).
Include units where helpful: For scientific code, including units like _nm (nanometers) or _px (pixels) in variable names prevents ambiguity.

III. Code Formatting & Structure
A consistent format and logical structure make code easier to read and debug.
Follow PEP 8: Use a linter or autoformatter to adhere to the standard Python style guide for spacing, line length, and indentation.
Logical Script Order: Structure scripts in a predictable sequence:
Imports: All imports at the top.
Global Parameters / Constants: A capitalized, clearly marked section for user settings.
Function Definitions: Group related functions together under section headers (e.g., Analysis Functions, Data Loading Functions).
Main Execution Block: The main script logic should be contained within a main() function and called from under an if __name__ == '__main__': block.

IV. Function & Docstring Design
Functions are the building blocks of a program and must be documented clearly.
Single Responsibility: Each function should do one thing well.
Functional Docstrings: Docstrings should be concise and explain the function's purpose, its parameters, and what it returns.
Python
def calculate_mean(numbers):
    """
    Calculates the arithmetic mean of a list of numbers.

    Args:
        numbers (list): A list of numeric values.

    Returns:
        float: The calculated mean, or float('nan') if the list is empty.
    """
    if not numbers:
        return float('nan') # Return NaN for empty lists to avoid errors.
    return sum(numbers) / len(numbers)



