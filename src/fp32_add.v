module fp32_add(
    input clk, 

    input [31:0] a,b,

    output reg [31:0] c
);
    

// not consider denormal, NaN, Inf,Zero

wire a_sign = a[31];
wire b_sign = b[31];

wire [7:0] a_exp = a[30:23];
wire [7:0] b_exp = b[30:23];

wire [23:0] a_mant = {1'b1,a[22:0]};
wire [23:0] b_mant = {1'b1,b[22:0]};

// exp max find


wire m_f = (a_exp >= b_exp ) ? 1'b1 : 1'b0;
wire [7:0] exp_max = (m_f) ? a_exp : b_exp;
wire [7:0] exp_m = (m_f) ?  a_exp-b_exp : b_exp-a_exp;
wire [4:0] exp_cut = (exp_m >= 24) 24 ? exp_m[4:0];
wire [23:0] shift_mant = (m_f) ?  b_mant>>exp_cut  : a_mant>>exp_cut;
wire [23:0] sel_mant = (m_f) ? a_mant : b_mant;
wire [24:0] add = shift_mant + sel_mant;
wire [22:0] final_val = (add[25]) ? add[23:1] : add[22:0];
wire [7:0] exp_final = (add[25]) ? exp_max+'b1 : exp_max;

wire sign = a_sign^b_sign;

always @(posedge clk ) begin
    c <= {sign,exp_final,final_val};
end



//using dsp for shift



endmodule