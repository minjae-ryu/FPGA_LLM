module fp32_add(
    input clk, 

    input [31:0] a,b,

    output [31:0] c
);
    

// not consider subzero, NaN, Inf

wire a_sign = a[31];
wire b_sign = b[31];

wire [7:0] a_exp = a[30:23];
wire [7:0] b_exp = b[30:23];



endmodule