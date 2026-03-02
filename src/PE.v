// q8_0 

// FPGA_TECHNOLOGY seven serise =1 ultrascale + = 2
module PE#
(
    parameter CORE_DELAY = 6
)

(
    input [63:0] 





);


// logic
// GEMM mode GEMV mode Softmax mode




PE_q8 mul_logic(
    .clk(clk),
    .delay(valid[2:0]),
    .pass(pass),
    .w(w),
    .a(a),
    .o(o)
);

scale_mul scale_logic(
    .clk(clk),
    .valid(valid[2:0]),
    .ws(ws),
    .as(as),
    .o_scale(o_scale)
);





endmodule








