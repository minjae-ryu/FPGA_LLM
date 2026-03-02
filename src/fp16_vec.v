`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/03/02 12:01:45
// Design Name: 
// Module Name: f16_vec
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module f16_vec #( 
    parameter integer OUT_W = 44
)
(
    input       clk,
    input       rst_n,
    input [5:0] ce, 

    input [16*8-1:0] in_a,in_b,

    output signed [OUT_W-1:0] o
);


genvar i;

wire [16-1:0] vec_a[0:8-1];
wire [16-1:0] vec_b[0:8-1];
wire signed [OUT_W-1:0] add_0[0:8-1];
generate
    for(i = 0; i < 8; i = i + 1)begin
        assign vec_a[i] = in_a[16*(i+1)-1:16*i];
        assign vec_b[i] = in_b[16*(i+1)-1:16*i];
        fp16_mul #(
            .OUT_W(OUT_W)
        ) fp16_uut(
            .clk(clk),
            .rst_n(rst_n),
            .ce(ce[2:0]),
            .in_a(vec_a[i]),
            .in_b(vec_b[i]),
            .o(add_0[i])
        );
    end
endgenerate


reg signed [OUT_W-1:0] add_1[0:4-1];

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        add_1[0] <= 'd0;
        add_1[1] <= 'd0;
        add_1[2] <= 'd0;
        add_1[3] <= 'd0;
    end else if(ce[3])begin
        add_1[0] <= add_0[0] + add_0[1];
        add_1[1] <= add_0[2] + add_0[3];
        add_1[2] <= add_0[4] + add_0[5];
        add_1[3] <= add_0[6] + add_0[7];
    end
end

reg signed [OUT_W-1:0] add_2[0:2-1];


always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        add_2[0] <= 'd0;
        add_2[1] <= 'd0;
    end else if(ce[4])begin
        add_2[0] <= add_1[0] + add_1[1];
        add_2[1] <= add_1[2] + add_1[3];
    end
end

reg signed [OUT_W-1:0] add_3;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        add_3 <= 'd0;
    end else if(ce[5])begin
        add_3 <= add_2[0] + add_2[1];
    end
end

assign o = add_3;

endmodule
