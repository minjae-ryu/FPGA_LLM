
//support fp16 dot product 

module PE_fp(
    input clk,
    input [16*8-1:0] a,b,
    output [31:0] out_fp
);

wire a_sign[8-1:0];
wire b_sign[8-1:0];

wire [5-1:0] a_exp[8-1:0];
wire [5-1:0] b_exp[8-1:0];

wire [10-1:0] a_mant[8-1:0];
wire [10-1:0] b_mant[8-1:0];

// find exp_max

genvar i;

generate
    for(i = 0; i < 8; i = i + 1)begin
        assign a_sign[i] = a[16*(i+1)-1];
        assign b_sign[i] = b[16*(i+1)-1];
        assign a_exp[i] = a[16*(i+1)-2:16*(i+1)-6];
        assign b_exp[i] = b[16*(i+1)-2:16*(i+1)-6];
        assign a_mant[i] = a[16*(i+1)-7:16*i];
        assign b_mant[i] = b[16*(i+1)-7:16*i];
    end
endgenerate

reg [5-1:0] max_st0 [4-1:0];
reg [5-1:0] max_st1 [2-1:0];
reg [5-1:0] max_st2;
//st0
always @(posedge clk ) begin
    max_st0[0] <= (a_exp[0] >= a_exp[1]) ? a_exp[0] : a_exp[1]
    max_st0[1] <= (a_exp[2] >= a_exp[3]) ? a_exp[2] : a_exp[3];
    max_st0[2] <= (a_exp[4] >= a_exp[5]) ? a_exp[4] : a_exp[5];
    max_st0[3] <= (a_exp[6] >= a_exp[7]) ? a_exp[6] : a_exp[7];    
end




//st1
always @(posedge clk ) begin
    max_st1[0] <= (max_st0[0] >= max_st0[1]) ? max_st0[0] : max_st0[1];
    max_st1[1] <= (max_st0[2] >= max_st0[3]) ? max_st0[2] : max_st0[3];    
end

//st2
always @(posedge clk ) begin
    max_st2 <= (max_st1[0] >= max_st1[1]) ? max_st1[0] : max_st1[1];
end




endmodule