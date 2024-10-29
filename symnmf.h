# ifndef HEADER_FILE
# define HEADER_FILE

double** sym_c(double **X, int num_of_elements, int d);
double** ddg_c(double **X, int num_of_elements, int d);
double** norm_c(double **X, int num_of_elements, int d);
double** symnmf_c(double **H, double **W, int k, int num_of_elements);
double** matrix_allocation(int num_of_rows, int num_of_cols);
void free_matrix(double **matrix);
void printMatrix(double **matrix, int num_of_rows,  int num_of_cols);

# endif