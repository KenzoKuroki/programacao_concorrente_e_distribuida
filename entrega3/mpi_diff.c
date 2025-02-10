#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 2000
#define T 1000
#define D 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new, int local_n, int rank, int numProc) {
    for (int t = 0; t < T; t++) {
        int previous = rank - 1;
        int next     = rank + 1;

        if (rank == 0) {
        // Processo 0: apenas troca com o próximo.
        MPI_Sendrecv(&C[local_n][0],N, MPI_DOUBLE,next, 0,
                     &C[local_n+1][0],N, MPI_DOUBLE,next, 1,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    } else if (rank == numProc - 1) {
        // Último processo: apenas troca com o anterior.
        MPI_Sendrecv(&C[1][0],N, MPI_DOUBLE,previous, 1,
                     &C[0][0],N, MPI_DOUBLE,previous, 0,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    } else {
        // Processos intermediários: troca com o anterior e com o seguinte.
        // Troca com o anterior:
        MPI_Sendrecv(&C[1][0],N, MPI_DOUBLE,previous, 1,
                     &C[0][0],N, MPI_DOUBLE,previous, 0,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        // Troca com o próximo:
        MPI_Sendrecv(&C[local_n][0],N, MPI_DOUBLE,next, 0,
                     &C[local_n+1][0],N, MPI_DOUBLE,next, 1,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

        for (int i = 1; i <= local_n; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T *
                    ((C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j])
                     / (DELTA_X * DELTA_X));
            }
        }

        double difmedio_local = 0.0;
        for (int i = 1; i <= local_n; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio_local += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        double difmedio_total = 0.0;
        MPI_Reduce(&difmedio_local, &difmedio_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && t % 100 == 0) {
            printf("Iteracao %d - diferenca=%g\n", t, difmedio_total / ((N-2)*(N-2)));
        }
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, numProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    int base = N / numProc;
    int resto = N % numProc;
    int local_n = (rank < resto) ? base + 1 : base;
    int inicio = 1 + ((rank < resto) ? rank*(base+1) : resto*(base+1) + (rank-resto)*base);
    int final = inicio + local_n - 1;

    /*Criando matrizes com N/numProc linhas, sendo
    adicionado linhas nos extremos para a comunicacao*/

    double **C = (double **)malloc((local_n + 2) * sizeof(double *));
    double **C_new = (double **)malloc((local_n + 2) * sizeof(double *));

    for (int i = 0; i < local_n + 2; i++) {
        C[i] = (double *)malloc(N * sizeof(double));
        C_new[i] = (double *)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    int center_y = (N / 2)-1;
    int center_x = center_y - inicio+1;

    if (center_y >= inicio && center_y <= final) {
        C[center_x][center_y] = 1.0;

    }
    diff_eq(C, C_new, local_n, rank, numProc);
    double conc_center = 0.0;
    if (center_y >= inicio && center_y <= final) {
        conc_center = C[center_x][center_y];
    }

    double *final_conc = NULL;
    if (rank == 0) final_conc = (double *)malloc(numProc * sizeof(double));
    MPI_Gather(&conc_center, 1, MPI_DOUBLE, final_conc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < numProc; i++) {
            if (final_conc[i] > 0) {
                printf("Concentração final no centro: %f\n", final_conc[i]);
                break;
            }
        }
        free(final_conc);
    }

    // Liberar memória
    for (int i = 0; i < local_n + 2; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}