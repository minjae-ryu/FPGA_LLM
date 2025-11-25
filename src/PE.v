// q8_0 

// FPGA_TECHNOLOGY seven serise =1 ultrascale + = 2
module PE#
(
    parameter CORE_DELAY = 6
)
    (
    input               clk,


    input [8*256-1:0]    act,
    input [6*256-1:0]    weight,

    input [8*16-1:0]      scale,




    output  [29:0]       out,

);



wire [6*16-1:0] w_vec[16-1:0];
wire [8*16-1:0] a_vec[16-1:0];
wire signed [18+8-1:0] o_vec[16-1:0];
wire signed [18+9-1:0] ad0[8-1:0];

reg signed [18+10-1:0] ad1 [4-1:0];
wire signed [18+11-1:0] ad2 [2-1:0];
reg signed [18+12-1:0] ad3 ;

genvar a;
generate
    for(a = 0; a < 16; a = a + 1)begin : Q6_K_MUL
    assign w_vec[a] = weight[(a+1)*6*16-1:a*6*16];
    assign a_vec[a] = act[(a+1)*8*16-1:a*8*16];
        PE_int PE_int_uut(
            .clk(clk),
            .a(a_vec[a]),
            .w(w_vec[a]),
            .scale(scale[(a+1)*8-1:a*8]),
            .o(o_vec[a])
        );
    end
endgenerate

genvar b;
generate
    for(b = 0; b < 8; b = b+1)begin
        assign ad0[b] = o_vec[b*2+1] + o_vec[b*2];
    end
endgenerate

always @(posedge clk ) begin
    ad1[0] <= ad0[0] + ad0[1];
    ad1[1] <= ad0[2] + ad0[3];
    ad1[2] <= ad0[4] + ad0[5];
    ad1[3] <= ad0[6] + ad0[7];
end

assign ad2[0] = ad1[0] + ad1[1];
assign ad2[1] = ad1[2] + ad1[3];

always @(posedge clk ) begin
    ad3 <= ad2[0] + ad2[1];
end

assign out = ad3;
endmodule








