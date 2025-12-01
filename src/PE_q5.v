module PE_q5(
    input           clk,
    input [5*32-1:0] w,
    input [8*32-1:0] a,

    output [18-1:0] o

);
    

wire signed [5-1:0] w_vec[32-1:0];
wire signed [8-1:0] a_vec[32-1:0];

reg signed  [5+8-1:0] mul[32-1:0];
genvar i;

generate
    for(i = 0; i < 32; i = i + 1)begin
        assign w_vec[i] = w[(i+1)*5-1:i*5];
        assign a_vec[i] = a[(i+1)*8-1:i*8];
        always @(posedge clk ) begin
            mul[i] <= w_vec[i]*a_vec[i];
        end
    end
endgenerate


wire signed [14-1:0] ad0[16-1:0];
wire signed [15-1:0] ad1[8-1:0];
reg  signed [16-1:0] ad2[4-1:0];
wire signed [17-1:0] ad3[2-1:0];
reg  signed [18-1:0] ad4;

genvar st0;

generate
    for(st0 = 0; st0 < 16; st0 = st0 + 1)begin
        assign ad0[st0] = mul[st0*2+1] + mul[st0*2];
    end
endgenerate

genvar st1;
generate
    for(st1 = 0; st1 < 8; st1 = st1 + 1)begin
        assign ad1[st1] = ad0[st1*2+1] + ad0[st1*2];
    end
endgenerate

genvar st2;
generate
    for(st2 = 0; st2 < 4; st2 = st2 + 1) begin
        always @(posedge clk ) begin
            ad2[st2] <= ad1[st2*2+1] + ad1[st2*2];
        end
    end
endgenerate

assign ad3[0] = ad2[0] + ad2[1];
assign ad3[1] = ad2[2] + ad2[3];


always @(posedge clk ) begin
    ad4 <= ad3[0] + ad3[1];
end

assign o = ad4;



endmodule