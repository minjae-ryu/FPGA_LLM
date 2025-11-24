module PE_int(
    input                   clk,

    input [8*16-1:0]        a
    input [6*16-1:0]        w,

    input signed [8-1:0]    scale,
    output signed [18+8-1:0]    o


)


wire signed [8-1:0] a_vec [16-1:0];
wire signed [6-1:0] w_vec [16-1:0];
reg signed [14-1:0] mul_vec [16-1:0];


genvar i;

generate
    for(i = 0; i < 16; i = i + 1)begin
        assign a_vec[i] = a[(i+1)*8-1:i*8];
        assign w_vec[i] = w[(i+1)*6-1:i*6];
        always @(posedge clk) begin
            mul_vec[i] <= a_vec[i] * w_vec[i];
        end
    end
endgenerate



wire signed [15-1:0] st0w_ad [8-1:0];

genvar j;

generate
    for(j = 0; j < 8; j = j + 1)begin
        assign st0w_ad[j] = mul_vec[j*2+1] + mul_vec[j*2];
    end
endgenerate

reg signed [16-1:0] st0r_ad [4-1:0];

always @(posedge clk) begin
    st0r_ad[0] <= st0w_ad[0] + st0w_ad[1];
    st0r_ad[1] <= st0w_ad[2] + st0w_ad[3];
    st0r_ad[2] <= st0w_ad[4] + st0w_ad[5];
    st0r_ad[3] <= st0w_ad[6] + st0w_ad[7];
end

wire signed [17-1:0] st1w_ad [2-1:0];

assign st1w_ad [0] = st0r_ad[0] + at0r_ad[1];
assign st1w_ad [1] = st0r_ad[2] + at0r_ad[3];

reg signed [18-1:0] st1r_ad;

always @(posedge clk) begin
    st1r_ad <= st1w_ad[0] + st1w_ad[1];
end


reg signed [18+8-1:0] output_mul;
reg signed [8-1:0] scale_d [2:0];

always @(posedge clk) begin
    scale_d[0] <= scale;
    scale_d[1] <= scale[0];
    scale_d[2] <= scale[1];   
    output_mul <= sclae[2] * st1r_ad;
end


assign o = output_mul;




endmodule 